from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
if not QDRANT_URL:
    st.error("QDRANT_URL is missing in .env file. Add it before running.")
    st.stop()

_METADATA_VECTOR_DIM = 1

try:
    qdrant = QdrantClient(url=QDRANT_URL)
except Exception as e:
    st.error(f"Unable to connect to Qdrant at {QDRANT_URL}: {e}")
    st.stop()


def ensure_metadata_collection(name: str):
    try:
        qdrant.get_collection(name)
    except Exception:
        # create collection with 1-d vectors (we'll store payloads, vector set to [0.0])
        try:
            qdrant.recreate_collection(
                collection_name=name,
                vectors_config=rest.VectorParams(size=_METADATA_VECTOR_DIM, distance=rest.Distance.COSINE),
            )
        except Exception as e:
            st.error(f"Could not create Qdrant collection '{name}': {e}")
            st.stop()

#ensure_metadata_collection("users")
#ensure_metadata_collection("tickets")
#ensure_metadata_collection("ticket_conversations")

# =============================
# USERS (persisted in Qdrant 'users' collection)
# =============================
def load_users_from_qdrant():
    """Load users from qdrant into session_state.users (map by email)."""
    try:
        users = {}
        # scroll returns (points, next_page)
        points, _ = qdrant.scroll(collection_name="users", limit=1000)
        for point in points:
            payload = point.payload or {}
            email = payload.get("email")
            if email:
                users[email] = payload
        st.session_state.users = users
    except Exception as e:
        st.session_state.users = {}
        st.warning(f"Could not load users from DB: {e}")

def create_ticket():
     """
    Create a placeholder ticket at greet time.
    Title/description will be filled on the first user message.
    """
    current_user_email = st.session_state.current_user
    user_obj = st.session_state.users.get(current_user_email, {})
    user_name = user_obj.get("name", "Unknown")

    # Correct UUID object (not string)
    ticket_uuid = uuid.uuid4()
    ticket_id = str(ticket_uuid)

    ticket_payload = {
        "ticket_id": ticket_id,
        "title": "",
        "user_id": current_user_email,
        "user_name": user_name,
        "description": "",
        "priority": 2,
        "status": "open",
        "urgency": "medium",
        "category": "general",
        "knowledge_base_id": "",
        "assigned_to": "agent_ai_01",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "resolved_at": "",
        "is_resolved": False
    }

    try:
        logger.info(f"create_ticket: creating ticket {ticket_id} for user {current_user_email}")
        qdrant.upsert(
            collection_name="tickets",
            points=[
                rest.PointStruct(
                    id=ticket_uuid,   # UUID object required
                    vector=[0.0],
                    payload=ticket_payload
                )
            ]
        )
        logger.info(f"create_ticket: upserted ticket {ticket_id}")
        initialize_ticket_conversation(ticket_id)

    except Exception as e:
        logger.exception("Failed to create ticket in DB")
        st.error(f"Failed to create ticket in DB: {e}")
        return None

    st.session_state.current_ticket_id = ticket_id
    st.session_state.ticket_created = True
    return ticket_payload

def get_all_tickets():
    tickets = []
    try:
        points, _ = qdrant.scroll(collection_name="tickets", limit=1000)
        for p in points:
            tickets.append(p.payload)
    except Exception as e:
        st.error(f"Failed to fetch tickets: {e}")
    return tickets

def update_ticket_metadata(ticket_id: str, updates: dict):
    """
    Update (partial) fields of the ticket with ticket_id.
    """
    try:
        points, _ = qdrant.scroll(collection_name="tickets", limit=1000)
        for p in points:
            payload = p.payload or {}
            if payload.get("ticket_id") == ticket_id:
                payload.update(updates)
                payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
                qdrant.upsert(
                    collection_name="tickets",
                    points=[rest.PointStruct(
                        id=p.id,
                        vector=[0.0],
                        payload=payload
                    )]
                )
                return payload
    except Exception as e:
        st.error(f"Failed to update ticket: {e}")
    return None

def save_user_to_qdrant(user_data: dict):
    """
    Save user to Qdrant:
      - Create a Qdrant-safe UUID as the point id
      - Keep the email inside the payload (so we can map by email on load)
    """
    try:
        # Ensure email present
        email = user_data.get("email")
        if not email:
            raise ValueError("user_data must include 'email'")

        # Use a UUID as the Qdrant point id (Qdrant requires int or UUID)
        point_id = str(uuid.uuid4())
        # Ensure payload contains the email (so load_users_from_qdrant can index by email)
        user_data_copy = dict(user_data)
        user_data_copy["qid"] = point_id
        # Upsert into Qdrant
        qdrant.upsert(
            collection_name="users",
            points=[
                rest.PointStruct(
                    id=point_id,
                    vector=[0.0],  # dummy vector for metadata collection
                    payload=user_data_copy,
                )
            ],
        )
    except Exception as e:
        # Show error but do not crash; caller should handle result
        st.error(f"Failed to save user to DB: {e}")
        raise

# initialize users from qdrant
#load_users_from_qdrant()

def retrieve_ticket_byid(ticketId):
    try:
        result = qdrant.retrieve(
            collection_name="tickets",
            ids=[uuid.UUID(ticket_id)]
        )
        return ticket_payload = result[0].payload if result else None
    except Exception as e:
        st.error(f"Ticket load failed: {e}")
        return "Error loading ticket.", False

# =============================
# COOKIE MANAGER
# =============================
cookie_manager = stx.CookieManager()
current_user_cookie = cookie_manager.get("current_user")
# If cookie exists and user exists in loaded users, mark authenticated
if current_user_cookie and current_user_cookie in st.session_state.users:
    st.session_state.authenticated = True
    st.session_state.current_user = current_user_cookie


def update_ticket_status(ticket_id: str, new_status: str):
    """
    Update only status (and mark resolved fields when appropriate).
    """
    updates = {"status": new_status}
    if new_status.lower() in ("resolved", "closed", "closed_by_user"):
        updates["is_resolved"] = True
        updates["resolved_at"] = datetime.utcnow().isoformat() + "Z"
    return update_ticket_metadata(ticket_id, updates)

def initialize_ticket_conversation(ticket_id):
    payload = {
        "ticket_id": ticket_id,
        "conversation": [],
        "events": []
    }
    try:
        logger.info(f"initialize_ticket_conversation: initializing conversation for ticket {ticket_id}")
        qdrant.upsert(
            collection_name="ticket_conversations",
            points=[rest.PointStruct(
                id=ticket_id,
                vector=[0.0],
                payload=payload
            )]
        )
    except Exception as e:
        logger.exception("Failed to initialize ticket conversation")
        st.error(f"Failed to initialize ticket conversation: {e}")

def get_ticket_conversation(ticket_id):
    try:
        points, _ = qdrant.scroll(collection_name="ticket_conversations", limit=1000)
        for p in points:
            payload = p.payload or {}
            if payload.get("ticket_id") == ticket_id:
                return payload
    except Exception as e:
        st.error(f"Failed to read ticket conversation: {e}")
    return None

def add_conversation_message(ticket_id, message_payload):
    """
    Append a message dict to the 'conversation' array of the ticket_conversations document.
    """
    try:
        points, _ = qdrant.scroll(collection_name="ticket_conversations", limit=1000)
        for p in points:
            payload = p.payload or {}
            if payload.get("ticket_id") == ticket_id:
                payload.setdefault("conversation", []).append(message_payload)
                logger.info(f"add_conversation_message: appending message to ticket {ticket_id}")
                qdrant.upsert(
                    collection_name="ticket_conversations",
                    points=[rest.PointStruct(
                        id=p.id,
                        vector=[0.0],
                        payload=payload
                    )]
                )
                logger.info(f"add_conversation_message: appended for ticket {ticket_id}")
                return payload
        # if not found, initialize and insert
        initialize_ticket_conversation(ticket_id)
        # append again
        payload = {
            "ticket_id": ticket_id,
            "conversation": [message_payload],
            "events": []
        }
        qdrant.upsert(
            collection_name="ticket_conversations",
            points=[rest.PointStruct(
                id=ticket_id,
                vector=[0.0],
                payload=payload
            )]
        )
        logger.info(f"add_conversation_message: created new conversation doc for ticket {ticket_id}")
        return payload
    except Exception as e:
        logger.exception("Failed to append conversation message")
        st.error(f"Failed to append conversation message: {e}")
        return None

def add_ticket_event(ticket_id, event_type, actor_type, actor_id, message):
    event_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "actor_type": actor_type,
        "actor_id": actor_id,
        "message": message
    }
    try:
        points, _ = qdrant.scroll(collection_name="ticket_conversations", limit=1000)
        for p in points:
            payload = p.payload or {}
            if payload.get("ticket_id") == ticket_id:
                payload.setdefault("events", []).append(event_payload)
                logger.info(f"add_ticket_event: appending event {event_type} to ticket {ticket_id}")
                qdrant.upsert(
                    collection_name="ticket_conversations",
                    points=[rest.PointStruct(
                        id=p.id,
                        vector=[0.0],
                        payload=payload
                    )]
                )                
                return payload
        # If not found, initialize doc with this event
        payload = {
            "ticket_id": ticket_id,
            "conversation": [],
            "events": [event_payload]
        }
        qdrant.upsert(
            collection_name="ticket_conversations",
            points=[rest.PointStruct(
                id=ticket_id,
                vector=[0.0],
                payload=payload
            )]
        )
        logger.info(f"add_ticket_event: created new conversation doc with event for ticket {ticket_id}")
        return payload
    except Exception as e:
        logger.exception("Failed to append ticket event")
        st.error(f"Failed to append ticket event: {e}")
        return None


def initialize_user_history(user_id, name, tier):
    """
    Initialize user history when a new user registers or first logs in.
    """
    history_payload = {
        "user_id": user_id,
        "name": name,
        "tier": tier.lower(),
        "last_login": datetime.utcnow().isoformat() + "Z",
        "recent_activity": {
            "payment_attempts": 0,
            "failed_payments": 0,
            "speed_tests": [],
            "logins": 1
        },
        "metrics": {
            "account_health": 100,
            "payment_success_rate": 100,
            "network_stability": 100
        },
        "past_tickets": []
    }
    
    try:
        # Generate a UUID for the point id
        point_id = str(uuid.uuid4())
        logger.info(f"initialize_user_history: creating history for {user_id} qid={point_id}")
        qdrant.upsert(
            collection_name="user_history",
            points=[rest.PointStruct(
                id=point_id,
                vector=[0.0],
                payload=history_payload
            )]
        )
        logger.info(f"initialize_user_history: upsert complete for {user_id} qid={point_id}")
        try:
            st.sidebar.success(f"Initialized user history for {user_id}")
        except Exception:
            pass
        return history_payload
    except Exception as e:
        logger.exception("Failed to initialize user history")
        st.error(f"Failed to initialize user history: {e}")
        return None


def get_user_history(user_id):
    """
    Retrieve user history from Qdrant.
    """
    try:
        logger.info(f"get_user_history: fetching history for {user_id}")
        points, _ = qdrant.scroll(collection_name="user_history", limit=1000)
        for p in points:
            payload = p.payload or {}
            if payload.get("user_id") == user_id:
                logger.info(f"get_user_history: found history for {user_id} (point id={p.id})")
                return payload
        logger.info(f"get_user_history: no history found for {user_id}")
        return None
    except Exception as e:
        logger.exception("Failed to retrieve user history")
        st.error(f"Failed to retrieve user history: {e}")
        return None


def update_user_history(user_id, updates):
    """
    Update user history with new data.
    """
    try:
        logger.info(f"update_user_history: updating history for {user_id} with updates keys={list(updates.keys())}")
        points, _ = qdrant.scroll(collection_name="user_history", limit=1000)
        for p in points:
            payload = p.payload or {}
            if payload.get("user_id") == user_id:
                # Deep merge updates
                for key, value in updates.items():
                    if isinstance(value, dict) and key in payload:
                        payload[key].update(value)
                    else:
                        payload[key] = value

                qdrant.upsert(
                    collection_name="user_history",
                    points=[rest.PointStruct(
                        id=p.id,
                        vector=[0.0],
                        payload=payload
                    )]
                )
                logger.info(f"update_user_history: upsert complete for {user_id} (point id={p.id})")
                try:
                    st.sidebar.success(f"Updated user history for {user_id}")
                except Exception:
                    pass
                return payload
        logger.info(f"update_user_history: no history point found to update for {user_id}")
        try:
            st.sidebar.info(f"No existing user_history entry found for {user_id}")
        except Exception:
            pass
        return None
    except Exception as e:
        logger.exception("Failed to update user history")
        st.error(f"Failed to update user history: {e}")
        return None


def calculate_metrics(user_id):
    """
    Calculate user metrics based on their activity and ticket history.
    """
    history = get_user_history(user_id)
    if not history:
        return
    
    past_tickets = history.get("past_tickets", [])
    total_tickets = len(past_tickets)
    resolved_tickets = sum(1 for t in past_tickets if t.get("resolved", False))
    
    # Calculate account health based on resolved tickets ratio
    if total_tickets > 0:
        resolution_rate = (resolved_tickets / total_tickets) * 100
        account_health = int(resolution_rate * 0.6 + 40)  # Base 40, up to 100
    else:
        account_health = 100
    
    # Network stability based on speed tests (if available)
    speed_tests = history.get("recent_activity", {}).get("speed_tests", [])
    if speed_tests:
        avg_speed = sum(speed_tests) / len(speed_tests)
        network_stability = min(100, int(avg_speed * 0.8))
    else:
        network_stability = 100
    
    # Payment success rate (placeholder for future payment integration)
    payment_attempts = history.get("recent_activity", {}).get("payment_attempts", 0)
    failed_payments = history.get("recent_activity", {}).get("failed_payments", 0)
    
    if payment_attempts > 0:
        payment_success_rate = int(((payment_attempts - failed_payments) / payment_attempts) * 100)
    else:
        payment_success_rate = 100
    
    metrics = {
        "account_health": account_health,
        "payment_success_rate": payment_success_rate,
        "network_stability": network_stability
    }
    
    update_user_history(user_id, {"metrics": metrics})

def add_speed_test(user_id, speed_mbps):
    """
    Add a network speed test result to user history.
    """
    history = get_user_history(user_id)
    if not history:
        return
    
    speed_tests = history.get("recent_activity", {}).get("speed_tests", [])
    speed_tests.append(round(speed_mbps, 2))
    
    # Keep only last 5 speed tests
    if len(speed_tests) > 5:
        speed_tests = speed_tests[-5:]
    
    updates = {
        "recent_activity": {
            **history.get("recent_activity", {}),
            "speed_tests": speed_tests
        }
    }
    update_user_history(user_id, updates)
    
    # Recalculate metrics after adding speed test
    calculate_metrics(user_id)

def display_user_history(user_id):
    """
    Display user history in the sidebar.
    """
    history = get_user_history(user_id)
    if not history:
        st.sidebar.warning("No history available")
        return
    
    st.sidebar.markdown("### 📊 User History")
    st.sidebar.markdown(f"**Name:** {history.get('name', 'Unknown')}")
    st.sidebar.markdown(f"**Tier:** {history.get('tier', 'N/A').capitalize()}")
    
    # Format last login date
    last_login = history.get('last_login', 'Never')
    if last_login != 'Never':
        try:
            login_date = datetime.fromisoformat(last_login.replace('Z', '+00:00'))
            last_login = login_date.strftime('%Y-%m-%d %H:%M')
        except:
            pass
    st.sidebar.markdown(f"**Last Login:** {last_login}")
    
    st.sidebar.markdown("#### Recent Activity")
    activity = history.get("recent_activity", {})
    st.sidebar.metric("Total Logins", activity.get("logins", 0))
    
    speed_tests = activity.get("speed_tests", [])
    if speed_tests:
        avg_speed = sum(speed_tests) / len(speed_tests)
        st.sidebar.metric("Avg Speed (Mbps)", f"{avg_speed:.1f}")
    
    st.sidebar.markdown("#### Health Metrics")
    metrics = history.get("metrics", {})
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Account Health", f"{metrics.get('account_health', 100)}%")
    with col2:
        st.metric("Network", f"{metrics.get('network_stability', 100)}%")
    
    st.sidebar.markdown("#### Past Tickets")
    past_tickets = history.get("past_tickets", [])
    if past_tickets:
        # Show last 3 tickets
        for ticket in past_tickets[-3:]:
            status = "✅" if ticket.get("resolved") else "⏳"
            issue = ticket.get("issue", "N/A")
            # Truncate long issue titles
            if len(issue) > 40:
                issue = issue[:40] + "..."
            st.sidebar.markdown(f"{status} **{issue}**")
    else:
        st.sidebar.info("No tickets yet")
    
    # Optional: Add speed test simulator button
    # st.sidebar.markdown("---")
    # if st.sidebar.button("🚀 Run Speed Test"):
    #     import random
    #     speed = random.uniform(50, 150)
    #     add_speed_test(user_id, speed)
    #     st.sidebar.success(f"Speed: {speed:.2f} Mbps recorded!")
    #     st.rerun()


def add_ticket_to_history(user_id, ticket_id, issue_title, resolved=False):
    """
    Add a ticket to user's past tickets.
    """
    history = get_user_history(user_id)
    if not history:
        return
    
    past_tickets = history.get("past_tickets", [])
    
    # Check if ticket already exists
    existing_ticket = None
    for i, ticket in enumerate(past_tickets):
        if ticket.get("ticket_id") == ticket_id:
            existing_ticket = i
            break
    
    ticket_entry = {
        "ticket_id": ticket_id,
        "issue": issue_title,
        "resolved": resolved
    }
    
    if existing_ticket is not None:
        # Update existing ticket
        past_tickets[existing_ticket] = ticket_entry
    else:
        # Add new ticket
        past_tickets.append(ticket_entry)
    
    update_user_history(user_id, {"past_tickets": past_tickets})


def record_login(user_id):
    """
    Record a user login event and update last_login timestamp.
    """
    history = get_user_history(user_id)
    
    if not history:
        # If history doesn't exist, initialize it
        user_data = st.session_state.users.get(user_id, {})
        history = initialize_user_history(
            user_id,
            user_data.get("name", "Unknown"),
            user_data.get("tier", "staff")
        )
    
    if history:
        updates = {
            "last_login": datetime.utcnow().isoformat() + "Z",
            "recent_activity": {
                **history.get("recent_activity", {}),
                "logins": history.get("recent_activity", {}).get("logins", 0) + 1
            }
        }
        update_user_history(user_id, updates)

def upsert_ticket_vector(ticket_id: str, text: str):
    """
    Store or update a semantic vector for a ticket in Qdrant.
    `text` should summarize the ticket (title + description, etc.).
    """
    try:
        emb_model = get_ticket_embedding_model()
        vector = emb_model.embed_query(text)

        qdrant.upsert(
            collection_name="ticket_vectors",
            points=[
                rest.PointStruct(
                    id=ticket_id,
                    vector=vector,
                    payload={
                        "ticket_id": ticket_id,
                        "text": text,
                    },
                )
            ],
        )
        logger.info(f"upsert_ticket_vector: stored vector for ticket {ticket_id}")
    except Exception as e:
        logger.exception("Failed to upsert ticket vector")

def search_similar_tickets(query: str, top_k: int = 3, score_threshold: float = 0.8):
    """
    Semantic search over past tickets by user query.
    Returns a list of Qdrant ScoredPoint objects.
    """
    try:
        emb_model = get_ticket_embedding_model()
        q_vec = emb_model.embed_query(query)

        results = qdrant.search(
            collection_name="ticket_vectors",
            query_vector=q_vec,
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold,
        )
        return results
    except Exception as e:
        logger.exception("Ticket vector search failed")
        return []
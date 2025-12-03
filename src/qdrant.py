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

        initialize_ticket_conversation(ticket_id)

    except Exception as e:
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
load_users_from_qdrant()

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
        qdrant.upsert(
            collection_name="ticket_conversations",
            points=[rest.PointStruct(
                id=ticket_id,
                vector=[0.0],
                payload=payload
            )]
        )
    except Exception as e:
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
                qdrant.upsert(
                    collection_name="ticket_conversations",
                    points=[rest.PointStruct(
                        id=p.id,
                        vector=[0.0],
                        payload=payload
                    )]
                )
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
        return payload
    except Exception as e:
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
        return payload
    except Exception as e:
        st.error(f"Failed to append ticket event: {e}")
        return None
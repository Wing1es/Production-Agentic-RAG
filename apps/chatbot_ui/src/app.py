import uuid
import requests
import logging
import streamlit as st
from chatbot_ui.core.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Ecommerce Assitant",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

session_id = get_session_id()

def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message
        }

    try:
        response = getattr(requests, method.lower())(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data
        
        return False, response_data
    
    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please try again later.")
        return False, {"message": "Connection error"}
    
    except requests.exceptions.Timeout:
        _show_error_popup("Timeout error. Please try again later.")
        return False, {"message": "Timeout error"}

    except requests.exceptions.RequestException as e:
        _show_error_popup(f"Error: {str(e)}")
        return False, {"message": str(e)}

def submit_feedback(feedback_type=None, feedback_text=""):

    def _feedback_score(feedback_type):
        if feedback_type == "positive":
            return 1
        elif feedback_type == "negative":
            return 0
        else:
            return None
    
    feedback_data = {
        "feedback_score": _feedback_score(feedback_type),
        "feedback_text": feedback_text,
        "trace_id": st.session_state.trace_id,
        "thread_id": session_id,
        "feedback_source_type": "api"
    }

    logger.info("Feedback data: %s", feedback_data)

    status, response = api_call("POST", f"{config.API_URL}/submit_feedback/", json=feedback_data)
    return status, response

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

if "used_context" not in st.session_state:
    st.session_state.used_context = []

if "latest_feedback" not in st.session_state:
    st.session_state.latest_feedback = None

if "show_feedback_box" not in st.session_state:
    st.session_state.show_feedback_box = False

if "feedback_submission_status" not in st.session_state:
    st.session_state.feedback_submission_status = None

if "trace_id" not in st.session_state:
    st.session_state.trace_id = None

with st.sidebar:
    suggestions_tab, = st.tabs(["Suggestions"])

    with suggestions_tab:
        if st.session_state.used_context:
            for idx, item in enumerate(st.session_state.used_context):
                st.caption(item.get("description", "No description"))
                if "image_url" in item:
                    st.image(item["image_url"], width=250)
                st.caption(f"Price: {item.get('price', 'N/A')} USD")
                st.markdown("---")
        else:
            st.caption("No suggestions yet")

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Add feedback buttons only for the latest assistant message (excluding the initial greeting)
        is_latest_assistant = (
            message["role"] == "assistant"
            and idx == len(st.session_state.messages) - 1
            and idx > 0
        )

        if is_latest_assistant:
            # Use Streamlit's built-in feedback component
            feedback_key = f"feedback_{len(st.session_state.messages)}"
            feedback_result = st.feedback("thumbs", key=feedback_key)

            # Handle feedback selection
            if feedback_result is not None:
                feedback_type = (
                    "positive" if feedback_result == 1 else "negative"
                )

                # Only submit if this is a new/different feedback
                if st.session_state.latest_feedback != feedback_type:
                    with st.spinner("Submitting feedback..."):
                        status, response = submit_feedback(
                            feedback_type=feedback_type
                        )

                        if status:
                            st.session_state.latest_feedback = feedback_type
                            st.session_state.feedback_submission_status = "success"
                            st.session_state.show_feedback_box = (
                                feedback_type == "negative"
                            )
                        else:
                            st.session_state.feedback_submission_status = "error"
                            st.error("Failed to submit feedback. Please try again.")

                    st.rerun()

            # Show feedback status message
            if (
                st.session_state.latest_feedback
                and st.session_state.feedback_submission_status == "success"
            ):
                if st.session_state.latest_feedback == "positive":
                    st.success("✅ Thank you for your positive feedback!")
                elif (
                    st.session_state.latest_feedback == "negative"
                    and not st.session_state.show_feedback_box
                ):
                    st.success("✅ Thank you for your feedback!")
            elif st.session_state.feedback_submission_status == "error":
                st.error("❌ Failed to submit feedback. Please try again.")

            # Show feedback text box if thumbs down was pressed
            if st.session_state.show_feedback_box:
                st.markdown("**Want to tell us more? (Optional)**")
                st.caption(
                    "Your negative feedback has already been recorded. "
                    "You can optionally provide additional details below."
                )

                # Text area for detailed feedback
                feedback_text = st.text_area(
                    "Additional feedback (Optional)",
                    key=f"feedback_text_{len(st.session_state.messages)}",
                    placeholder="Please describe what was wrong with this response...",
                    height=100,
                )

                # Send additional feedback button
                col_send, col_spacer, col_close = st.columns([3, 5, 2])

                with col_send:
                    if st.button(
                        "Send Additional Details",
                        key=f"send_additional_{len(st.session_state.messages)}",
                    ):
                        if feedback_text.strip():  # Only send if there's actual text
                            with st.spinner("Submitting additional feedback..."):
                                status, response = submit_feedback(
                                    feedback_text=feedback_text
                                )

                                if status:
                                    st.success(
                                        "✅ Thank you! Your additional feedback has been recorded."
                                    )
                                    st.session_state.show_feedback_box = False
                                else:
                                    st.error(
                                        "❌ Failed to submit additional feedback. Please try again."
                                    )
                        else:
                            st.warning(
                                "Please enter some feedback text before submitting."
                            )

                        st.rerun()

                with col_close:
                    if st.button(
                        "Close",
                        key=f"close_feedback_{len(st.session_state.messages)}",
                    ):
                        st.session_state.show_feedback_box = False
                        st.rerun()

if prompt := st.chat_input("Ask Anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        success, response_data = api_call("POST", f"{config.API_URL}/rag/", json={
            "query": prompt,
            "thread_id": session_id
        })

        if success:
            reply = response_data.get("answer", "")
            used_context = response_data.get("used_context", [])
            trace_id = response_data.get("trace_id")

            st.session_state.used_context = used_context
            st.session_state.trace_id = trace_id

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            error_msg = response_data.get("message") or "Unknown error occurred"
            st.error(error_msg)
    
    st.rerun()
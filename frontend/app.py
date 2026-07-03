import streamlit as st
import requests
from PIL import Image
import os

# Configure page
st.set_page_config(
    page_title="Math LLM - Hindi Tutor",
    page_icon="🧮",
    layout="centered"
)

API_URL = os.getenv("API_URL", "http://localhost:8000/solve")

st.title("🧮 Math LLM for Hindi Medium Students")
st.markdown("### आपका AI गणित शिक्षक (Your AI Math Tutor)")
st.write("Enter a mathematics question in Hindi, and the AI will provide a step-by-step solution.")

# Optional OCR / Image upload
st.markdown("#### प्रश्न पूछें (Ask a Question)")
tab1, tab2 = st.tabs(["✍️ Text Input", "📷 Upload Image (Mock OCR)"])

question_text = ""

with tab1:
    question_text = st.text_area("अपना प्रश्न यहाँ लिखें:", height=100, placeholder="उदाहरण: 2x + 3 = 7, तो x का मान क्या है?")

with tab2:
    uploaded_file = st.file_uploader("प्रश्न की फोटो अपलोड करें:", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Question', use_column_width=True)
        st.info("Note: OCR is currently in mock mode. The text will not be extracted from the image in this version.")
        # Mock OCR output
        if not question_text:
            question_text = "2x + 3 = 7, तो x का मान क्या है?"
            st.text_area("OCR Result (Mock):", value=question_text, height=100, disabled=True)

if st.button("हल करें (Solve)", type="primary"):
    if not question_text.strip():
        st.warning("कृपया कोई प्रश्न दर्ज करें। (Please enter a question.)")
    else:
        with st.spinner("AI सोच रहा है... (AI is thinking...)"):
            try:
                response = requests.post(API_URL, json={"question": question_text}, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("समाधान मिल गया! (Solution found!)")
                    st.markdown("### उत्तर (Answer):")
                    
                    # Display steps
                    for i, step in enumerate(data.get("steps", [])):
                        st.markdown(f"**Step {i+1}:** {step}")
                        
                    st.markdown("---")
                    st.markdown(f"**पूर्ण समाधान (Full Solution):**\n\n{data.get('solution')}")
                    st.caption(f"Confidence Score: {data.get('confidence', 0.0):.2f}")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"API से कनेक्ट नहीं हो सका। कृपया सुनिश्चित करें कि बैकएंड ({API_URL}) चल रहा है।")
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown("Developed with Mistral-7B and QLoRA for Hindi Medium Students.")

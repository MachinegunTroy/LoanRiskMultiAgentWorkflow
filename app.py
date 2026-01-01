import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="Loan Risk AI", page_icon="🏦", layout="wide")
st.title("🏦 GenAI Loan Risk Assessment System")
st.markdown("### Powered by Hybrid-Agent Architecture (Flan-T5 + Logic Guardrails)")

# --- 2. CACHED MODEL LOADING (Run Once) ---
@st.cache_resource(show_spinner="Downloading AI Brain (Flan-T5)...")
def load_model():
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_length=256,
        repetition_penalty=1.2, 
        do_sample=True,
        temperature=0.3
    )
    return HuggingFacePipeline(pipeline=pipe)

# --- 3. CACHED DATA LOADING ---
@st.cache_data
def load_data():
    try:
        # Looking for data in the 'data/' folder as per your repo structure
        df_c = pd.read_csv('data/credit.csv', dtype=str)
        df_a = pd.read_csv('data/account.csv', dtype=str)
        df_p = pd.read_csv('data/pr.csv', dtype=str)
        return df_c, df_a, df_p
    except FileNotFoundError:
        return None, None, None

# Load Resources
llm = load_model()
df_c, df_a, df_p = load_data()

# --- 4. AGENT SYSTEM CLASS ---
class LoanMultiAgentSystem:
    def __init__(self, llm, df_c, df_a, df_p):
        self.llm, self.df_c, self.df_a, self.df_p = llm, df_c, df_a, df_p

    def run_assessment(self, customer_id):
        # AGENT 1: DATA RETRIEVAL
        cred = self.df_c[self.df_c['ID'] == str(customer_id)]
        acct = self.df_a[self.df_a['ID'] == str(customer_id)]
        
        profile = {
            "Name": cred.iloc[0]['Name'],
            "Credit Score": int(cred.iloc[0]['Credit Score']),
            "Nationality": acct.iloc[0]['Nationality'],
            "Account Status": acct.iloc[0]['Account Status'],
            "PR Status": "N/A"
        }
        if profile['Nationality'] == "Non-Singaporean":
            pr_rec = self.df_p[self.df_p['ID'] == str(customer_id)]
            profile['PR Status'] = pr_rec.iloc[0]['PR Status'] if not pr_rec.empty else "false"
        
        return profile

    def analyze_risk(self, profile):
        # AGENT 2: POLICY EXPERT (Hybrid Logic)
        score = profile['Credit Score']
        status = profile['Account Status'].lower()
        
        # Risk Logic
        risk = "Medium"
        if 300 <= score <= 674:
            risk = "High" if status == "delinquent" else "Medium"
        elif 675 <= score <= 749:
            risk = "Medium"
        elif 750 <= score <= 850:
            risk = "Medium" if status == "delinquent" else "Low"

        # Rate Logic
        if risk == "Low": rate = "3.175%"
        elif risk == "Medium": rate = "4.885%"
        else: rate = "6.325%"
        
        return risk, rate

    def draft_email(self, profile, decision, guidance, risk, rate):
        # AGENT 3: COMMUNICATOR (AI)
        prompt = ChatPromptTemplate.from_template(
            """You are a polite bank officer. Write a 2-sentence email to {name}. 
            State that their loan is {decision} because {guidance}. 
            Mention Risk Level: {risk} and Rate: {rate}."""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "name": profile['Name'], "decision": decision, 
            "guidance": guidance, "risk": risk, "rate": rate
        })

# --- 5. MAIN APP UI ---
if df_c is not None:
    system = LoanMultiAgentSystem(llm, df_c, df_a, df_p)
    
    # Sidebar for Inputs
    st.sidebar.header("🔍 Customer Lookup")
    all_ids = df_c['ID'].tolist()
    labels = {id: f"{id} - {df_c[df_c['ID']==id]['Name'].iloc[0]}" for id in all_ids}
    
    selected_id = st.sidebar.selectbox("Select Customer:", options=all_ids, format_func=lambda x: labels[x])

    if st.sidebar.button("🚀 Run Assessment"):
        # DISPLAY PROGRESS
        with st.status("🤖 AI Agents at work...", expanded=True) as status:
            st.write("📂 Agent 1: Retrieving financial records...")
            profile = system.run_assessment(selected_id)
            
            st.write("⚖️ Agent 2: Analyzing risk policies & calculating rates...")
            risk, rate = system.analyze_risk(profile)
            
            st.write("🛡️ Agent 3: verifying regulatory compliance...")
            is_non_sg = profile['Nationality'] == "Non-Singaporean"
            has_pr = str(profile['PR Status']).lower() == "true"
            
            if is_non_sg and not has_pr:
                decision = "REJECTED"
                guidance = "Permanent Residency is required"
            else:
                decision = "APPROVED"
                guidance = "you meet our eligibility criteria"
                
            st.write("📧 Generative AI: Drafting decision email...")
            email = system.draft_email(profile, decision, guidance, risk, rate)
            
            status.update(label="✅ Assessment Complete!", state="complete", expanded=False)

        # DISPLAY RESULTS
        st.divider()
        st.subheader(f"Decision: {decision}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Credit Score", profile['Credit Score'])
        col2.metric("Risk Level", risk)
        col3.metric("Interest Rate", rate)
        col4.metric("PR Status", profile['PR Status'])
        
        st.info(f"**Drafted Email:**\n\n{email}")

else:
    st.error("❌ Data files not found! Please check your 'data/' folder in GitHub.")

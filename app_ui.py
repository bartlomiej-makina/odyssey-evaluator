import streamlit as st
import pandas as pd
import requests
import json
import os
from groq import Groq
from typing import Optional, Dict, Any
import time
import io

# Set page config
st.set_page_config(
    page_title="OdysseyAI Q&A Evaluator",
    page_icon="🎯",
    layout="wide"
)

class QuestionAnswerEvaluatorUI:
    def __init__(self):
        self.groq_client = None
        self.workspace_id = None
        self.conversation_id = None
        self.agent_id = None
        self.odyssey_api_key = None
        self.user_id = None  # Add user_id as configurable
        self.parameter_config = None  # Add this for agent parameters
        self.environment = "production"  # Add environment selection
        self.base_url = "https://app.odysseyai.ai/api"  # Default to production
    
    def set_environment(self, environment: str):
        """Set the environment and update base URL"""
        self.environment = environment
        if environment == "staging":
            self.base_url = "https://app.stage.odysseyai.ai/api"
        else:
            self.base_url = "https://app.odysseyai.ai/api"
    
    def setup_apis(self, groq_api_key, odyssey_api_key):
        """Initialize API clients"""
        try:
            self.groq_client = Groq(api_key=groq_api_key)
            self.odyssey_api_key = odyssey_api_key
            return True, "APIs initialized successfully"
        except Exception as e:
            return False, f"Error initializing APIs: {e}"
    
    def create_conversation(self, workspace_id: str, conversation_name: str = None) -> tuple[bool, str, str]:
        """Create a new conversation in OdysseyAI"""
        try:
            url = f"{self.base_url}/conversations"
            
            headers = {
                'x-api-key': f'{self.odyssey_api_key}',
                'userId': self.user_id,
                'Content-Type': 'application/json'
            }
            
            # Generate a default conversation name if not provided
            if not conversation_name:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conversation_name = f"Q&A Evaluation - {timestamp}"
            
            payload = {
                'workspaceId': workspace_id,
                'conversationName': conversation_name
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                response_data = response.json()
                conversation_id = response_data.get('conversationId') or response_data.get('id')
                
                if conversation_id:
                    return True, conversation_id, f"Conversation '{conversation_name}' created successfully"
                else:
                    return False, None, f"No conversation ID in response: {response_data}"
            else:
                return False, None, f"Failed to create conversation: {response.status_code} - {response.text}"
                
        except Exception as e:
            return False, None, f"Error creating conversation: {str(e)}"
    
    def call_external_api(self, question: str = None, agent_inputs: dict = None) -> str:
        """Call OdysseyAI API to get answer for the question."""
        try:
            url = f"{self.base_url}/chat/message"
            
            headers = {
                'x-api-key': f'{self.odyssey_api_key}',
                'userId': self.user_id,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'workspaceId': self.workspace_id,
                'conversationId': self.conversation_id,
                'disableMemory': True
            }
            
            if self.agent_id:
                payload['agentId'] = self.agent_id
                
                if self.agent_id in ['dqp-agent', 'agenframe']:
                    payload['message'] = question
                elif agent_inputs:
                    payload['agentInputs'] = agent_inputs
                else:
                    payload['message'] = question
            else:
                payload['message'] = question
            
            response = requests.post(url, headers=headers, json=payload, timeout=320)
            
            if response.status_code == 200:
                response_data = response.json()
                
                if 'data' in response_data and 'response' in response_data['data']:
                    return response_data['data']['response']
                elif 'message' in response_data:
                    return response_data['message']
                elif 'content' in response_data:
                    return response_data['content']
                elif 'response' in response_data:
                    return response_data['response']
                else:
                    return str(response_data)
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"API Error: {str(e)}"
    
    def evaluate_with_groq(self, question: str, expected_answer: str, api_answer: str, mode: str = "comparison") -> Dict[str, Any]:
        """Use Groq to evaluate the accuracy of the API answer"""
        try:
            if mode == "comparison":
                # Original comparison-based evaluation
                evaluation_prompt = f"""
                Question: {question}
                Expected Answer: {expected_answer}
                OdysseyAI Answer: {api_answer}
                
                Please evaluate the OdysseyAI answer against the expected answer using the following comprehensive criteria:

                **Primary Evaluation Criteria (Weighted):**
                1. **Completeness (20%)**: Does the answer include all key source information needed to address the prompt?
                2. **Clarity & Cohesion (15%)**: Is the answer clear, logical, and appropriately styled?
                3. **Nuance & Specificity (15%)**: Are entities, qualifiers (dates, locations), and relationships accurate?

                **Detailed Assessment Areas:**
                - **Explicit Question Relevance**: Does it correctly address the specific prompt asked?
                - **Accuracy & Hallucination Prevention**: Does it maintain factual accuracy without introducing unsupported information?
                - **Entity and Conceptual Alignment**: Are domain-specific terms, entities, and concepts used accurately?
                - **Contextual Boundary Adherence**: Is the answer grounded in and faithful to the source context?
                - **Contextual Nuance Preservation**: Are temporal constraints, qualifiers, and ambiguities handled accurately?
                - **Contextual Omission Detection**: Are all crucial contextual elements incorporated?
                - **Scope & Detail Alignment**: Does the answer match the required level of detail and scope?

                **Scoring Scale Reference:**
                5 = Excellent (Fully meets criteria, no errors)
                4 = Good (Minor issues, core purpose intact)
                3 = Fair (Moderate issues, noticeable flaws)
                2 = Poor (Significant issues, substantially compromised)
                1 = Unacceptable (Major errors, unusable)

                You MUST respond with valid JSON in exactly this format:
                {{
                    "score": [number from 0-100],
                    "is_correct": "[yes or no]",
                    "explanation": "[detailed explanation covering key criteria]",
                    "differences": "[specific differences and areas of concern]",
                    "criteria_breakdown": {{
                        "completeness": [1-5],
                        "clarity_cohesion": [1-5],
                        "nuance_specificity": [1-5],
                        "relevance": [1-5],
                        "accuracy": [1-5]
                    }}
                }}

                is_correct should be "yes" if the OdysseyAI answer contains information that is in the expected answer.

                Do not include any text before or after the JSON.
                """
            else:  # criteria mode
                # Criteria-based evaluation without expected answer comparison
                evaluation_prompt = f"""
                Question: {question}
                OdysseyAI Answer: {api_answer}
                
                Please evaluate the OdysseyAI answer based on the following comprehensive criteria (no comparison to expected answer):

                **Primary Evaluation Criteria (Weighted):**
                1. **Completeness (20%)**: Does the answer thoroughly address all aspects of the question?
                2. **Clarity & Cohesion (15%)**: Is the answer clear, well-structured, and easy to understand?
                3. **Nuance & Specificity (15%)**: Are specific details, qualifiers, and relationships appropriately addressed?
                4. **Relevance (25%)**: Does the answer directly address the question asked?
                5. **Quality & Accuracy (25%)**: Is the answer factually sound and free from obvious errors?

                **Detailed Assessment Areas:**
                - **Question Relevance**: Does it directly and completely address the specific question asked?
                - **Information Quality**: Is the information provided accurate, relevant, and well-sourced?
                - **Structure & Clarity**: Is the answer well-organized and easy to follow?
                - **Completeness**: Does it cover all important aspects of the question?
                - **Specificity**: Are appropriate details and examples provided where needed?
                - **Professional Quality**: Is the answer professional and appropriate for the context?

                **Scoring Scale Reference:**
                5 = Excellent (Fully meets criteria, high quality)
                4 = Good (Minor issues, overall strong response)
                3 = Fair (Moderate issues, acceptable but could be improved)
                2 = Poor (Significant issues, substantially flawed)
                1 = Unacceptable (Major problems, inadequate response)

                You MUST respond with valid JSON in exactly this format:
                {{
                    "score": [number from 0-100],
                    "quality_rating": "[excellent/good/fair/poor/unacceptable]",
                    "explanation": "[detailed explanation covering key criteria and overall assessment]",
                    "strengths": "[specific strengths of the answer]",
                    "areas_for_improvement": "[specific areas that could be improved]",
                    "criteria_breakdown": {{
                        "completeness": [1-5],
                        "clarity_cohesion": [1-5],
                        "nuance_specificity": [1-5],
                        "relevance": [1-5],
                        "accuracy": [1-5]
                    }}
                }}

                Do not include any text before or after the JSON.
                """
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": evaluation_prompt}],
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.0,
                max_tokens=4000,
            )
            
            evaluation_text = chat_completion.choices[0].message.content.strip()
            
            try:
                if '{' in evaluation_text and '}' in evaluation_text:
                    start = evaluation_text.find('{')
                    end = evaluation_text.rfind('}') + 1
                    json_text = evaluation_text[start:end]
                    evaluation = json.loads(json_text)
                else:
                    evaluation = json.loads(evaluation_text)
                
                # Handle different response formats based on mode
                if mode == "comparison":
                    # Validate and clean the evaluation data for comparison mode
                    if 'is_correct' in evaluation:
                        evaluation['is_correct'] = evaluation['is_correct'].lower().strip()
                        if evaluation['is_correct'] not in ['yes', 'no']:
                            evaluation['is_correct'] = 'no'
                    else:
                        evaluation['is_correct'] = 'yes' if evaluation.get('score', 0) >= 70 else 'no'
                    
                    # Ensure differences field exists
                    if 'differences' not in evaluation:
                        evaluation['differences'] = 'No differences specified'
                else:  # criteria mode
                    # Handle quality rating for criteria mode
                    if 'quality_rating' not in evaluation:
                        score = evaluation.get('score', 50)
                        if score >= 90:
                            evaluation['quality_rating'] = 'excellent'
                        elif score >= 80:
                            evaluation['quality_rating'] = 'good'
                        elif score >= 60:
                            evaluation['quality_rating'] = 'fair'
                        elif score >= 40:
                            evaluation['quality_rating'] = 'poor'
                        else:
                            evaluation['quality_rating'] = 'unacceptable'
                    
                    # Ensure strengths and areas_for_improvement exist
                    if 'strengths' not in evaluation:
                        evaluation['strengths'] = 'No specific strengths identified'
                    if 'areas_for_improvement' not in evaluation:
                        evaluation['areas_for_improvement'] = 'No specific improvements identified'
                
                # Common validation for both modes
                if 'score' not in evaluation or not isinstance(evaluation['score'], (int, float)):
                    evaluation['score'] = 50
                
                # Ensure criteria_breakdown exists and has valid values
                if 'criteria_breakdown' not in evaluation:
                    evaluation['criteria_breakdown'] = {
                        'completeness': 3,
                        'clarity_cohesion': 3,
                        'nuance_specificity': 3,
                        'relevance': 3,
                        'accuracy': 3
                    }
                else:
                    # Validate each criteria score
                    for criteria in ['completeness', 'clarity_cohesion', 'nuance_specificity', 'relevance', 'accuracy']:
                        if criteria not in evaluation['criteria_breakdown']:
                            evaluation['criteria_breakdown'][criteria] = 3
                        elif not isinstance(evaluation['criteria_breakdown'][criteria], (int, float)):
                            evaluation['criteria_breakdown'][criteria] = 3
                        elif evaluation['criteria_breakdown'][criteria] < 1 or evaluation['criteria_breakdown'][criteria] > 5:
                            evaluation['criteria_breakdown'][criteria] = max(1, min(5, evaluation['criteria_breakdown'][criteria]))
                
                # Ensure explanation field exists
                if 'explanation' not in evaluation:
                    evaluation['explanation'] = 'No explanation provided'
                
                return evaluation
                
            except json.JSONDecodeError:
                import re
                score_match = re.search(r'score["\s:]*(\d+)', evaluation_text, re.IGNORECASE)
                extracted_score = int(score_match.group(1)) if score_match else 50
                
                # Default response based on mode
                if mode == "comparison":
                    correct_match = re.search(r'(yes|no)', evaluation_text, re.IGNORECASE)
                    extracted_correct = correct_match.group(1).lower() if correct_match else 'no'
                    
                    return {
                        'score': extracted_score,
                        'is_correct': extracted_correct,
                        'explanation': evaluation_text[:300] + "..." if len(evaluation_text) > 300 else evaluation_text,
                        'differences': 'Unable to parse detailed differences',
                        'criteria_breakdown': {
                            'completeness': 3,
                            'clarity_cohesion': 3,
                            'nuance_specificity': 3,
                            'relevance': 3,
                            'accuracy': 3
                        }
                    }
                else:  # criteria mode
                    return {
                        'score': extracted_score,
                        'quality_rating': 'fair',
                        'explanation': evaluation_text[:300] + "..." if len(evaluation_text) > 300 else evaluation_text,
                        'strengths': 'Unable to parse specific strengths',
                        'areas_for_improvement': 'Unable to parse specific improvements',
                        'criteria_breakdown': {
                            'completeness': 3,
                            'clarity_cohesion': 3,
                            'nuance_specificity': 3,
                            'relevance': 3,
                            'accuracy': 3
                        }
                    }
                
        except Exception as e:
            # Default error response based on mode
            if mode == "comparison":
                return {
                    'score': 0,
                    'is_correct': 'no',
                    'explanation': f'Error during evaluation: {str(e)}',
                    'differences': 'Evaluation failed',
                    'criteria_breakdown': {
                        'completeness': 1,
                        'clarity_cohesion': 1,
                        'nuance_specificity': 1,
                        'relevance': 1,
                        'accuracy': 1
                    }
                }
            else:  # criteria mode
                return {
                    'score': 0,
                    'quality_rating': 'unacceptable',
                    'explanation': f'Error during evaluation: {str(e)}',
                    'strengths': 'Evaluation failed',
                    'areas_for_improvement': 'Evaluation failed',
                    'criteria_breakdown': {
                        'completeness': 1,
                        'clarity_cohesion': 1,
                        'nuance_specificity': 1,
                        'relevance': 1,
                        'accuracy': 1
                    }
                }
    
    def fetch_available_agents(self) -> list:
        """Fetch all available agents from OdysseyAI API"""
        try:
            url = f"{self.base_url}/agents"
            
            headers = {
                'x-api-key': f'{self.odyssey_api_key}',
                'userId': self.user_id,
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                response_data = response.json()
                
                if 'data' in response_data:
                    agents = response_data['data']
                    active_agents = [agent for agent in agents if agent.get('active', False)]
                    return active_agents
                else:
                    return []
            else:
                return []
                
        except Exception as e:
            st.error(f"Error fetching agents: {e}")
            return []

    def get_agent_details(self, agent_id: str) -> dict:
        """Get detailed information about a specific agent"""
        try:
            agents = self.fetch_available_agents()
            for agent in agents:
                if agent.get('agentid') == agent_id:
                    return agent
            return None
        except Exception as e:
            st.error(f"Error fetching agent details: {e}")
            return None
    
    def configure_agent_parameters(self, agent_details: dict, df: pd.DataFrame) -> dict:
        """Configure agent input parameters in Streamlit UI"""
        input_params = agent_details.get('inputparameters', {})
        
        if not input_params:
            return None
        
        st.subheader("🔧 Agent Parameter Configuration")
        
        # Extract parameter names based on data structure
        if isinstance(input_params, dict):
            param_names = list(input_params.keys())
        elif isinstance(input_params, list):
            param_names = input_params
        else:
            st.warning(f"Unexpected parameter format: {input_params}")
            return None
        
        st.write(f"**Agent '{agent_details.get('rootAgentName', 'Unknown')}' requires the following parameters:**")
        
        for param_name in param_names:
            if isinstance(input_params, dict) and isinstance(input_params[param_name], str):
                st.write(f"• **{param_name}**: {input_params[param_name]}")
            else:
                st.write(f"• **{param_name}**")
        
        parameter_config = {}
        
        # Create configuration for each parameter
        for param_name in param_names:
            st.write(f"**Configure parameter: {param_name}**")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                param_type = st.radio(
                    f"How to provide '{param_name}':",
                    ["Fixed value", "Map to column"],
                    key=f"param_type_{param_name}"
                )
            
            with col2:
                if param_type == "Fixed value":
                    value = st.text_input(
                        f"Fixed value for '{param_name}':",
                        key=f"param_value_{param_name}"
                    )
                    if value:
                        parameter_config[param_name] = {'type': 'fixed', 'value': value}
                else:
                    column_name = st.selectbox(
                        f"Column for '{param_name}':",
                        ["Select column..."] + list(df.columns),
                        key=f"param_column_{param_name}"
                    )
                    if column_name != "Select column...":
                        parameter_config[param_name] = {'type': 'mapped', 'column': column_name}
            
            st.divider()
        
        # Validate that all parameters are configured
        if len(parameter_config) == len(param_names):
            st.success("✅ All parameters configured!")
            return parameter_config
        else:
            missing_params = set(param_names) - set(parameter_config.keys())
            st.warning(f"⚠️ Please configure all parameters. Missing: {', '.join(missing_params)}")
            return None
    
    def get_agent_inputs_for_row(self, row, parameter_config: dict) -> dict:
        """Get agent inputs for a specific row based on parameter configuration"""
        if not parameter_config:
            return None
        
        agent_inputs = {}
        for param_name, config in parameter_config.items():
            if config['type'] == 'fixed':
                agent_inputs[param_name] = config['value']
            elif config['type'] == 'mapped':
                agent_inputs[param_name] = str(row[config['column']])
        
        return agent_inputs

def main():
    st.title("🎯 OdysseyAI Question-Answer Evaluator")
    st.markdown("Upload an Excel file and evaluate Q&A pairs using OdysseyAI and Groq")
    
    evaluator = QuestionAnswerEvaluatorUI()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Environment Selection
        st.subheader("🌍 Environment")
        environment = st.selectbox(
            "Select Environment",
            ["production", "staging"],
            index=0,
            help="Production: app.odysseyai.ai | Staging: app.stage.odysseyai.ai"
        )
        evaluator.set_environment(environment)
        st.info(f"Using: {evaluator.base_url}")
        
        # API Keys
        st.subheader("API Keys")
        groq_api_key = st.text_input("Groq API Key", type="password")
        odyssey_api_key = st.text_input("OdysseyAI API Key", type="password")
        
        # OdysseyAI Configuration
        st.subheader("OdysseyAI Settings")
        workspace_id = st.text_input("Workspace ID")
        user_id = st.text_input("User ID")
        
        # Optional conversation name
        conversation_name = st.text_input(
            "Conversation Name (optional)", 
            placeholder="Leave empty for auto-generated name"
        )
        
        # Initialize APIs
        if groq_api_key and odyssey_api_key and user_id:
            success, message = evaluator.setup_apis(groq_api_key, odyssey_api_key)
            if success:
                st.success("✅ APIs initialized")
                evaluator.workspace_id = workspace_id
                evaluator.user_id = user_id
            else:
                st.error(message)
    
    # Main content area
    col1, col2 = st.columns([3, 2])  # Changed from [2, 1] to [3, 2] for better balance
    
    with col1:
        st.header("📁 File Upload")
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows")
                
                # Show preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head())
                
                # Evaluation Mode Selection
                st.subheader("🔄 Evaluation Mode")
                evaluation_mode = st.radio(
                    "Select evaluation mode:",
                    ["comparison", "criteria"],
                    format_func=lambda x: {
                        "comparison": "🔍 Comparison Mode - Compare against expected answers",
                        "criteria": "📊 Criteria-based Mode - Evaluate based on quality criteria only"
                    }[x],
                    help="Comparison mode requires both question and answer columns. Criteria mode only needs questions."
                )
                
                if evaluation_mode == "comparison":
                    st.info("💡 **Comparison Mode**: Evaluates answers by comparing them against expected/correct answers from your data")
                else:
                    st.info("💡 **Criteria-based Mode**: Evaluates answers based on quality criteria without comparing to expected answers")
                
                # Column selection - Use more compact layout
                st.subheader("🔧 Column Configuration")
                if evaluation_mode == "comparison":
                    col_left, col_right = st.columns(2)
                    with col_left:
                        question_col = st.selectbox("Question Column", df.columns)
                    with col_right:
                        answer_col = st.selectbox("Expected Answer Column", df.columns)
                else:
                    question_col = st.selectbox("Question Column", df.columns)
                    answer_col = None
                    st.info("No answer column needed for criteria-based evaluation")
                
                # Agent selection
                if evaluator.odyssey_api_key and workspace_id:
                    st.subheader("🤖 Agent Selection")
                    
                    agents = evaluator.fetch_available_agents()
                    
                    # Add agenframe as a special option
                    agent_options = ["No agent (default chat)", "agenframe (special agent)"] + [
                        f"{agent.get('rootAgentName', agent.get('agentid', 'Unknown'))} ({agent.get('agentid')})"
                        for agent in agents
                    ]
                    
                    selected_agent_idx = st.selectbox("Select Agent", range(len(agent_options)), format_func=lambda x: agent_options[x])
                    
                    if selected_agent_idx == 0:
                        evaluator.agent_id = None
                        evaluator.parameter_config = None
                    elif selected_agent_idx == 1:
                        # agenframe selected
                        evaluator.agent_id = "agenframe"
                        evaluator.parameter_config = None
                        st.success("✅ Using agenframe (message-based agent)")
                        st.info("This agent uses message-based input (no parameter configuration needed)")
                    else:
                        # Regular agent selected (adjust index for the added agenframe option)
                        agent_idx = selected_agent_idx - 2
                        evaluator.agent_id = agents[agent_idx].get('agentid')
                        
                        # Show agent details in an expander to save space
                        selected_agent = agents[agent_idx]
                        with st.expander(f"ℹ️ Agent Details: {selected_agent.get('rootAgentName', 'Unknown')}"):
                            st.write(f"**Type:** {selected_agent.get('agenttype', 'Unknown')}")
                            st.write(f"**Description:** {selected_agent.get('description', 'No description')}")
                        
                        # Configure agent parameters if needed
                        if evaluator.agent_id and evaluator.agent_id not in ['dqp-agent', 'agenframe']:
                            agent_details = evaluator.get_agent_details(evaluator.agent_id)
                            if agent_details and agent_details.get('inputparameters'):
                                with st.expander("🔧 Agent Parameter Configuration", expanded=True):
                                    st.write("**Debug Info:**")
                                    st.write(f"Input parameters: {agent_details.get('inputparameters')}")
                                    st.write(f"Type: {type(agent_details.get('inputparameters'))}")
                                    
                                    evaluator.parameter_config = evaluator.configure_agent_parameters(agent_details, df)
                            else:
                                evaluator.parameter_config = None
                        else:
                            evaluator.parameter_config = None
                
                # Process button - only show if all required configs are set
                can_process = (evaluator.groq_client and evaluator.odyssey_api_key and 
                             workspace_id and evaluator.user_id and question_col and 
                             (answer_col is not None or evaluation_mode == "criteria"))
                
                # Additional check for agent parameters
                if evaluator.agent_id and evaluator.agent_id not in ['dqp-agent', 'agenframe']:
                    agent_details = evaluator.get_agent_details(evaluator.agent_id)
                    if agent_details and agent_details.get('inputparameters') and not evaluator.parameter_config:
                        st.warning("⚠️ Please configure all agent parameters first")
                        can_process = False
                
                if can_process:
                    if st.button("🚀 Start Evaluation", type="primary"):
                        # Create conversation first
                        st.info("Creating conversation...")
                        success, conversation_id, message = evaluator.create_conversation(
                            workspace_id, 
                            conversation_name if conversation_name.strip() else None
                        )
                        
                        if not success:
                            st.error(f"❌ Failed to create conversation: {message}")
                            st.stop()
                        
                        evaluator.conversation_id = conversation_id
                        st.success(f"✅ {message}")
                        st.info(f"Conversation ID: {conversation_id}")
                        
                        # Process the file
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Add new columns based on evaluation mode
                        df['OdysseyAI_Answer'] = ''
                        df['Accuracy_Score'] = 0
                        df['Evaluation_Explanation'] = ''
                        df['API_Status'] = ''
                        
                        # Add criteria breakdown columns (common to both modes)
                        df['Completeness_Score'] = 0
                        df['Clarity_Cohesion_Score'] = 0
                        df['Nuance_Specificity_Score'] = 0
                        df['Relevance_Score'] = 0
                        df['Accuracy_Criteria_Score'] = 0
                        
                        # Add mode-specific columns
                        if evaluation_mode == "comparison":
                            df['Is_Correct'] = ''
                            df['Differences'] = ''
                        else:  # criteria mode
                            df['Quality_Rating'] = ''
                            df['Strengths'] = ''
                            df['Areas_for_Improvement'] = ''
                        
                        # Show processing configuration in an expander to save space
                        with st.expander("ℹ️ Processing Configuration", expanded=True):
                            st.write(f"• Evaluation Mode: {evaluation_mode.title()}")
                            st.write(f"• Workspace ID: {workspace_id}")
                            st.write(f"• User ID: {evaluator.user_id}")
                            st.write(f"• Conversation ID: {conversation_id}")
                            if evaluator.agent_id:
                                st.write(f"• Agent ID: {evaluator.agent_id}")
                                if evaluator.parameter_config:
                                    st.write("• Parameter Configuration:")
                                    for param_name, config in evaluator.parameter_config.items():
                                        if config['type'] == 'fixed':
                                            st.write(f"  - {param_name}: Fixed value '{config['value']}'")
                                        else:
                                            st.write(f"  - {param_name}: Mapped to column '{config['column']}'")
                            else:
                                st.write("• Using default chat (no agent)")
                        
                        # Process each row
                        for index, row in df.iterrows():
                            status_text.text(f"Processing row {index + 1}/{len(df)}")
                            progress_bar.progress((index + 1) / len(df))
                            
                            # Get expected answer for comparison mode (or None for criteria mode)
                            expected_answer = str(row[answer_col]) if evaluation_mode == "comparison" else None
                            
                            # Get agent inputs for this specific row
                            agent_inputs = None
                            question = None
                            
                            if evaluator.parameter_config:
                                # Using agent with parameters - no message field
                                agent_inputs = evaluator.get_agent_inputs_for_row(row, evaluator.parameter_config)
                                # For evaluation purposes, use the question column value
                                question = str(row[question_col])
                            else:
                                # Using message-based approach
                                question = str(row[question_col])
                            
                            # Call OdysseyAI API
                            api_answer = evaluator.call_external_api(question, agent_inputs)
                            df.at[index, 'OdysseyAI_Answer'] = api_answer
                            
                            if api_answer.startswith("API Error:"):
                                df.at[index, 'API_Status'] = "Failed"
                                df.at[index, 'Accuracy_Score'] = 0
                                df.at[index, 'Evaluation_Explanation'] = "Could not evaluate due to API error"
                                
                                # Set mode-specific error values
                                if evaluation_mode == "comparison":
                                    df.at[index, 'Is_Correct'] = "N/A"
                                    df.at[index, 'Differences'] = "N/A"
                                else:  # criteria mode
                                    df.at[index, 'Quality_Rating'] = "N/A"
                                    df.at[index, 'Strengths'] = "N/A"
                                    df.at[index, 'Areas_for_Improvement'] = "N/A"
                            else:
                                df.at[index, 'API_Status'] = "Success"
                                
                                # Evaluate with Groq using the selected mode
                                evaluation = evaluator.evaluate_with_groq(question, expected_answer, api_answer, evaluation_mode)
                                
                                # Common fields for both modes
                                df.at[index, 'Accuracy_Score'] = evaluation.get('score', 0)
                                df.at[index, 'Evaluation_Explanation'] = evaluation.get('explanation', '')
                                
                                # Update criteria breakdown columns
                                df.at[index, 'Completeness_Score'] = evaluation['criteria_breakdown']['completeness']
                                df.at[index, 'Clarity_Cohesion_Score'] = evaluation['criteria_breakdown']['clarity_cohesion']
                                df.at[index, 'Nuance_Specificity_Score'] = evaluation['criteria_breakdown']['nuance_specificity']
                                df.at[index, 'Relevance_Score'] = evaluation['criteria_breakdown']['relevance']
                                df.at[index, 'Accuracy_Criteria_Score'] = evaluation['criteria_breakdown']['accuracy']
                                
                                # Mode-specific fields
                                if evaluation_mode == "comparison":
                                    df.at[index, 'Is_Correct'] = evaluation.get('is_correct', 'no')
                                    df.at[index, 'Differences'] = evaluation.get('differences', '')
                                else:  # criteria mode
                                    df.at[index, 'Quality_Rating'] = evaluation.get('quality_rating', 'fair')
                                    df.at[index, 'Strengths'] = evaluation.get('strengths', '')
                                    df.at[index, 'Areas_for_Improvement'] = evaluation.get('areas_for_improvement', '')
                            
                            time.sleep(2)  # Rate limiting
                        
                        status_text.text("✅ Processing complete!")
                        
                        # Show conversation link
                        conversation_link = f"https://app.odysseyai.ai/workspace/{workspace_id}/{conversation_id}"
                        st.success(f"🔗 [View conversation in OdysseyAI]({conversation_link})")
                        
                        # Show results in an expandable section
                        with st.expander("📊 View Results", expanded=True):
                            st.dataframe(df)
                        
                        # Summary statistics
                        st.subheader("📈 Summary Statistics")
                        successful_calls = len(df[df['API_Status'] == 'Success'])
                        failed_calls = len(df[df['API_Status'] == 'Failed'])
                        
                        if successful_calls > 0:
                            avg_score = df[df['API_Status'] == 'Success']['Accuracy_Score'].mean()
                            high_accuracy = len(df[(df['Accuracy_Score'] >= 80) & (df['API_Status'] == 'Success')])
                            low_accuracy = len(df[(df['Accuracy_Score'] < 50) & (df['API_Status'] == 'Success')])
                            
                            if evaluation_mode == "comparison":
                                correct_answers = len(df[(df['Is_Correct'] == 'yes') & (df['API_Status'] == 'Success')])
                                incorrect_answers = len(df[(df['Is_Correct'] == 'no') & (df['API_Status'] == 'Success')])
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Success Rate", f"{successful_calls}/{len(df)}")
                                with col2:
                                    st.metric("Correct Answers", f"{correct_answers}/{successful_calls}")
                                with col3:
                                    st.metric("Average Score", f"{avg_score:.1f}/100")
                                with col4:
                                    st.metric("High Accuracy (≥80)", f"{high_accuracy}")
                                
                                # Additional statistics for comparison mode
                                with st.expander("📋 Detailed Statistics"):
                                    st.write(f"• Failed API calls: {failed_calls}/{len(df)}")
                                    st.write(f"• Correct answers: {correct_answers}/{successful_calls} ({correct_answers/successful_calls*100:.1f}%)")
                                    st.write(f"• Incorrect answers: {incorrect_answers}/{successful_calls} ({incorrect_answers/successful_calls*100:.1f}%)")
                                    st.write(f"• Low Accuracy (<50): {low_accuracy} rows")
                            else:  # criteria mode
                                # Count quality ratings
                                excellent = len(df[(df['Quality_Rating'] == 'excellent') & (df['API_Status'] == 'Success')])
                                good = len(df[(df['Quality_Rating'] == 'good') & (df['API_Status'] == 'Success')])
                                fair = len(df[(df['Quality_Rating'] == 'fair') & (df['API_Status'] == 'Success')])
                                poor = len(df[(df['Quality_Rating'] == 'poor') & (df['API_Status'] == 'Success')])
                                unacceptable = len(df[(df['Quality_Rating'] == 'unacceptable') & (df['API_Status'] == 'Success')])
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Success Rate", f"{successful_calls}/{len(df)}")
                                with col2:
                                    st.metric("Excellent/Good", f"{excellent + good}/{successful_calls}")
                                with col3:
                                    st.metric("Average Score", f"{avg_score:.1f}/100")
                                with col4:
                                    st.metric("High Quality (≥80)", f"{high_accuracy}")
                                
                                # Additional statistics for criteria mode
                                with st.expander("📋 Quality Distribution"):
                                    st.write(f"• Failed API calls: {failed_calls}/{len(df)}")
                                    st.write(f"• Excellent: {excellent} ({excellent/successful_calls*100:.1f}%)")
                                    st.write(f"• Good: {good} ({good/successful_calls*100:.1f}%)")
                                    st.write(f"• Fair: {fair} ({fair/successful_calls*100:.1f}%)")
                                    st.write(f"• Poor: {poor} ({poor/successful_calls*100:.1f}%)")
                                    st.write(f"• Unacceptable: {unacceptable} ({unacceptable/successful_calls*100:.1f}%)")
                                    st.write(f"• Low Quality (<50): {low_accuracy} rows")
                        else:
                            st.error("❌ No successful API calls. Please check your configuration.")
                        
                        # Download button
                        output = io.BytesIO()
                        df.to_excel(output, index=False)
                        output.seek(0)
                        
                        st.download_button(
                            label="📥 Download Results",
                            data=output.getvalue(),
                            file_name=f"odyssey_evaluated_{evaluation_mode}_{uploaded_file.name}",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    if not (evaluator.groq_client and evaluator.odyssey_api_key and workspace_id and evaluator.user_id):
                        st.warning("⚠️ Please configure all API settings first")
                    elif evaluation_mode == "comparison" and not answer_col:
                        st.warning("⚠️ Please select an answer column for comparison mode")
                    elif evaluator.agent_id and evaluator.agent_id not in ['dqp-agent', 'agenframe']:
                        agent_details = evaluator.get_agent_details(evaluator.agent_id)
                        if agent_details and agent_details.get('inputparameters') and not evaluator.parameter_config:
                            st.warning("⚠️ Please configure all agent parameters first")
            
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    with col2:
        # Use tabs to organize instructions more compactly
        inst_tab1, inst_tab2, inst_tab3 = st.tabs(["📝 Steps", "🔄 Modes", "⚙️ Setup"])
        
        with inst_tab1:
            st.markdown("""
            **Quick Start:**
            1. Configure APIs in sidebar
            2. Upload Excel file
            3. Choose evaluation mode
            4. Select columns & agent
            5. Start evaluation
            """)
        
        with inst_tab2:
            st.markdown("""
            **🔍 Comparison Mode:**
            - Compare against expected answers
            - Needs question + answer columns
            - Shows accuracy & differences
            
            **📊 Criteria Mode:**
            - Quality-based evaluation only
            - Needs question column only
            - Shows ratings & suggestions
            """)
        
        with inst_tab3:
            with st.expander("📋 Requirements"):
                st.markdown("""
                - Excel file with question column
                - Groq API key
                - OdysseyAI API key + Workspace/User ID
                """)
            
            with st.expander("🤖 Agent Types"):
                st.markdown("""
                - **Message-based**: Use questions directly
                - **Parameter-based**: Need configuration
                - **agenframe**: Special dropdown option
                """)
            
            with st.expander("🌍 Environments"):
                st.markdown("""
                - **Production**: app.odysseyai.ai
                - **Staging**: app.stage.odysseyai.ai
                """)
        
        # Add some spacing
        st.write("")
        
        # Status indicator
        if evaluator.groq_client and evaluator.odyssey_api_key:
            st.success("✅ APIs Ready")
        else:
            st.warning("⚠️ Configure APIs first")

if __name__ == "__main__":
    main() 
# Save this file as streamlit_app.py

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
# Removed unused StateGraph, END imports as we are manually controlling flow
from typing import Dict, List, Optional, Union, TypedDict, Literal, Annotated
import os
import time # Need to import time

# -----------------------
# 💡 Define State (Added target_language)
# -----------------------
class SDLCState(TypedDict):
    stage: Annotated[str, "Current stage of the SDLC"]
    user_input: str
    target_language: Optional[str] # <<< ADDED: To store the desired code language
    user_stories: Optional[str]
    design_docs: Optional[str]
    code: Optional[str]
    test_cases: Optional[str]
    decision: Optional[Literal["approved", "feedback", "failed", "passed"]]
    feedback: Optional[str]
    history: list # Optional: Can be used to display full history if needed

# -----------------------
# ⚙️ LLM Setup (Groq - Use Streamlit secrets for API key ideally)
# -----------------------
# For local testing, you can keep it like this, but for deployment use secrets
# Make sure the API key is available as an environment variable or use st.secrets

# Assuming 'llm' is defined elsewhere in your application
llm = None  # Placeholder for the language model


# -----------------------
# 🔧 Nodes (Functions updated for multi-language support)
# -----------------------

def display_output(title, content, language='markdown'):
    """Helper function to display content nicely, with language support for code."""
    st.subheader(f"📄 {title}")
    # Determine language for st.code, default to python if not specific or markdown
    code_language = language.lower() if language else 'plaintext' # Use plaintext if unknown

    # Heuristic to decide if it's code vs markdown
    is_likely_code = False
    if isinstance(content, str):
         # Check common code keywords, structure, or if title suggests code/tests
        if "code" in title.lower() or "test" in title.lower():
            is_likely_code = True
        elif "```" in content or "def " in content or "import " in content or "class " in content or "function " in content or "{" in content:
             is_likely_code = True

    if is_likely_code:
        st.code(content, language=code_language, line_numbers=True)
    else:
        st.markdown(content) # Use markdown for descriptions, stories, etc.
    st.divider()

def clean_llm_code_output(content: str, language: str = None) -> str:
    """Cleans common LLM code block artifacts."""
    content = content.strip()
    # Remove ```language, ```, etc.
    language_tag_start = f"```{language.lower() if language else ''}".strip()
    if language and content.startswith(language_tag_start):
        content = content[len(language_tag_start):]
    elif content.startswith("```"):
         content = content[3:]

    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


# --- Node Functions (Updated Prompts) ---



def generate_user_stories(state: SDLCState):
    global llm
    # This function doesn't depend on the target language
    prompt = f"""
    You are a skilled Product Owner assistant tasked with converting the following user requirements into well-formed and comprehensive user stories. Your goal is to produce user stories that are specific, measurable, achievable, relevant, and time-bound (SMART), and that capture the essence of the user needs.

    Based ONLY on the user requirements below, generate 3-5 user stories in the following format:
    'As a [type of user], I want to [perform an action] so that [achieve a specific outcome/benefit]'.

    For each user story, also include a brief list of **acceptance criteria** that would confirm the story has been implemented correctly. Format the acceptance criteria as bullet points under each user story, starting with 'Acceptance Criteria:'.

    Consider different types of users who might interact with the system and try to cover the core functionalities and potentially some important non-functional aspects if implied by the requirements (e.g., performance, security, usability).

    Requirements:
    {state['user_input']}
    """
    with st.spinner("Generating Advanced User Stories..."):
        if llm:
            result = llm.invoke([HumanMessage(content=prompt)])
            content = result.content
            state["user_stories"] = content
            state["stage"] = "Product Owner Review"
            state["history"].append(("generate_user_stories", content)) # Keep history
            
            return state
        else:
            st.warning("LLM object is not initialized. Cannot generate user stories.")
            return state


def product_owner_review(state: SDLCState):
    global llm
    # This function doesn't depend on the target language
    prompt = f"""
    You are an experienced Product Owner reviewing the following user stories to ensure they are well-formed, comprehensive, and effectively capture the user needs.

    Evaluate each user story based on the following criteria:

    1.  **Format:** Does each story follow the standard format: 'As a [type of user], I want to [perform an action] so that [achieve a specific outcome/benefit]'?
    2.  **Clarity:** Is the user story easy to understand? Is the user type, action, and benefit clearly articulated?
    3.  **Completeness:** Does the user story capture a complete piece of functionality from the user's perspective?
    4.  **Value:** Is the benefit to the user clear and meaningful?
    5.  **Acceptance Criteria:** Are the acceptance criteria provided for each user story clear, measurable, and do they adequately define when the story is done? Do they align with the user story itself?

    Respond in one of two ways:

    1.  **Approval:** If all user stories are well-formed, clear, complete, provide clear value, and have adequate acceptance criteria, respond with ONLY:
        `approved`

    2.  **Feedback:** If any of the user stories need improvement based on the criteria above, respond with ONLY:
        `feedback: [provide specific and actionable suggestions for improvement. For each user story that needs work, clearly indicate the story and the specific issues (e.g., "User Story 1: The benefit is not clear.", "User Story 2: The acceptance criteria are missing or not measurable.", "User Story 3: The format is incorrect."). Be concise but provide enough detail for the author to understand the necessary changes.]`

    User Stories:
    {state['user_stories']}
    """
    with st.spinner("Product Owner AI Reviewing User Stories..."):
        if llm:
            result = llm.invoke([HumanMessage(content=prompt)])
            content = result.content.strip().lower()
            if content.startswith("approved"):
                state["decision"] = "approved"
                state["feedback"] = None
            elif content.startswith("feedback:"):
                state["decision"] = "feedback"
                state["feedback"] = content.split("feedback:", 1)[1].strip()
            else:
                st.warning(f"PO Review LLM sent unexpected response: '{content}'. Assuming feedback is needed.")
                state["decision"] = "feedback"
                state["feedback"] = f"LLM response unclear, please manually review and revise: '{content}'"
            state["history"].append(("product_owner_review", result.content)) # Keep history
            
            return state
        else:
            st.warning("LLM object is not initialized. Cannot review user stories.")
            return state




def revise_user_stories(state: SDLCState):
    global llm
    # This function doesn't depend on the target language
    feedback = state.get('feedback')
    original_stories = state.get('user_stories')
    original_input = state.get('user_input')

    if not feedback:
        st.info("No feedback provided for user stories. Skipping revision.")
        state["stage"] = "Product Owner Review"
        return state

    prompt = f"""
    You are a skilled Product Owner assistant tasked with revising the user stories based *only* on the feedback provided below. Your goal is to incorporate the suggestions while ensuring all user stories adhere to the standard format: 'As a [type of user], I want to [perform an action] so that [achieve a specific outcome/benefit]'.

    Pay close attention to any feedback regarding clarity, completeness, value, and the acceptance criteria associated with each user story. Ensure that the revisions address all the points mentioned in the feedback.

    **Feedback:**
    {feedback}

    **Original User Stories:**
    {original_stories}

    **Original User Requirements (for context):**
    {original_input}

    Return ONLY the complete set of revised user stories, ensuring each story and its acceptance criteria (if any) are correctly formatted and address the feedback.
    """
    with st.spinner("Revising User Stories based on feedback..."):
        if llm:
            result = llm.invoke([HumanMessage(content=prompt)])
            content = result.content
            state["user_stories"] = content
            state["decision"] = None
            state["feedback"] = None
            state["stage"] = "Product Owner Review"
            state["history"].append(("revise_user_stories", content)) # Keep history
            return state
        else:
            st.warning("LLM object is not initialized. Cannot revise user stories.")
            return state



def create_design_docs(state: SDLCState):
    global llm
    language = state.get('target_language', 'the target language')
    prompt = f"""
    You are an experienced software architect tasked with creating comprehensive design documents based on provided user stories. Your goal is to produce clear, concise, and actionable documentation that will guide the development team.

    Based ONLY on the approved user stories below, generate detailed design documents including:

    1. **Functional Design:**
       - Key features and their functionalities.
       - Detailed user flows for each key feature, including steps and potential variations.
       - High-level description of the user interface (UI) and user experience (UX) considerations.

    2. **Technical Design:**
       - **System Architecture Diagram:** Create a textual representation of the main components and their interactions. Use a format like Markdown to represent boxes (components) and arrows (interactions). Clearly label each component and the nature of the interaction.
       - Main components and modules with clear responsibilities.
       - Detailed data models, including entity relationships and data types (if applicable).
       - Key technologies, frameworks, and libraries that are suitable for implementing this application in {language}. Justify your choices and suggest specific options where relevant.
       - API design considerations (if the application involves APIs). Outline the key endpoints, request/response structures (briefly).
       - Database design considerations, including the type of database and rationale.
       - Deployment considerations (briefly outline potential deployment strategies).
       - Security considerations (mention key security aspects to be addressed).

    User Stories:
    {state['user_stories']}

    Ensure that the design is scalable, maintainable, and aligns with best practices for software development.
    """
    with st.spinner("Creating Design Documents..."):
        if llm:
            result = llm.invoke([HumanMessage(content=prompt)])
            content = result.content
            state["design_docs"] = content
            state["stage"] = "Design Review"
            state["history"].append(("create_design_docs", content)) # Keep history
            return state
        else:
            st.warning("LLM object is not initialized. Cannot create design documents.")
            return state


def design_review(state: SDLCState):
    global llm
    language = state.get('target_language', 'the specified language')
    prompt = f"""
    You are a highly experienced Senior Software Architect leading a design review. Your task is to critically evaluate the provided design documents based on the following criteria, ensuring they align with best practices and the provided user stories.

    **Review Criteria:**

    1.  **Functional Design Completeness & Clarity:**
        - Are all key features from the user stories addressed?
        - Are the user flows detailed, covering all necessary steps and potential variations?
        - Is the high-level UI/UX description sensible and user-friendly?

    2.  **Technical Design Soundness & Completeness:**
        - **System Architecture:** Is the proposed architecture clear, well-defined, and scalable? Does the textual diagram accurately represent the system? Are the component interactions logical?
        - **Component Responsibilities:** Are the responsibilities of each component/module clearly defined and appropriate?
        - **Data Models:** Are the data models comprehensive and well-structured? Do they accurately represent the data needed for the application?
        - **Technology Choices:** Are the suggested technologies, frameworks, and libraries suitable for implementing the application in {language}? Are the justifications sound? Are specific options suggested where appropriate?
        - **API Design (if applicable):** Are the outlined API endpoints and request/response structures logical and well-defined? Do they meet the needs of the functional design?
        - **Database Design:** Is the chosen database type appropriate for the application's needs? Is the rationale clear? Are key database considerations (e.g., schema, indexing) mentioned?
        - **Deployment Strategy:** Is the outlined deployment strategy feasible and appropriate for the application's scale and requirements?
        - **Security Considerations:** Are key security aspects relevant to the application mentioned and addressed at a high level?

    3.  **Alignment with User Stories:** Does the design fully address all the requirements outlined in the user stories? Are there any discrepancies or missing functionalities?

    4.  **Scalability & Maintainability:** Does the design consider potential future growth and the ease of maintaining the codebase? Are there any obvious design choices that might hinder scalability or maintainability?

    5.  **Best Practices:** Does the design adhere to generally accepted software development best practices and architectural patterns?

    **Instructions:**

    Respond in one of two ways:

    1.  **Approval:** If the design documents are well-thought-out, comprehensive, clearly articulated, and address all the review criteria adequately, respond with ONLY:
        `approved`

    2.  **Feedback:** If there are areas for improvement, respond with ONLY:
        `feedback: [provide specific and actionable suggestions for improving the design documents. For each point of feedback, clearly indicate the area of concern (e.g., "Functional Design: User flow for X is unclear," "Technical Design: Consider using Y framework for better scalability," "Alignment with User Stories: Feature Z from the user stories is not explicitly addressed"). Be concise but provide enough detail for the author to understand the necessary changes.]`

    Design Docs:
    {state['design_docs']}

    User Stories:
    {state['user_stories']}
    """
    with st.spinner("AI Reviewing Design Documents..."):
        if llm:
            result = llm.invoke([HumanMessage(content=prompt)])
            content = result.content.strip().lower()
            if content.startswith("approved"):
                state["decision"] = "approved"
                state["feedback"] = None
            elif content.startswith("feedback:"):
                state["decision"] = "feedback"
                state["feedback"] = content.split("feedback:", 1)[1].strip()
            else:
                st.warning(f"Design Review LLM sent unexpected response: '{content}'. Assuming feedback is needed.")
                state["decision"] = "feedback"
                state["feedback"] = f"LLM response unclear, please manually review and revise: '{content}'"
            state["history"].append(("design_review", result.content)) # Keep history
            return state
        else:
            st.warning("LLM object is not initialized. Cannot review design documents.")
            return state



def revise_design_docs(state: SDLCState):
    global llm
    language = state.get('target_language', 'the target language')
    prompt = f"""
    You are a highly skilled software architect tasked with revising the design documents based *only* on the feedback provided below. Your goal is to incorporate the suggestions while maintaining the overall structure of the design document (Functional Design and Technical Design) and ensuring the proposed changes are suitable for implementation in {language}.

    **Feedback:**
    {state.get('feedback', 'No feedback provided.')}

    **Original Design Documents:**
    {state['design_docs']}

    **Original User Stories (for context):**
    {state['user_stories']}

    Return ONLY the complete revised design documents, ensuring all sections (Functional Design and Technical Design) are present and updated according to the feedback. Do not include any additional explanations or markdown fences.
    """
    with st.spinner("Revising Design Documents based on feedback..."):
        if llm and state.get('feedback'):
            result = llm.invoke([HumanMessage(content=prompt)])
            content = result.content
            state["design_docs"] = content
            state["decision"] = None
            state["feedback"] = None
            state["stage"] = "Design Review"
            state["history"].append(("revise_design_docs", content)) # Keep history
            return state
        elif not state.get('feedback'):
            st.info("No feedback provided. Skipping design document revision.")
            state["stage"] = "Design Review" # Still go back to review
            state["history"].append(("revise_design_docs", "No feedback provided. Skipping design document revision."))
            return state
        
        else:
            st.warning("LLM object is not initialized. Cannot revise design documents.")
            return state




def generate_code(state: SDLCState):
    global llm
    language = state.get('target_language')
    if not language:
        st.error("Target language not set in state. Cannot generate code.")
        state['stage'] = 'User Input' # Go back if language missing? Or handle differently
        return state # Stop this path

    prompt = f"""
    You are an expert software developer specializing in {language}. Your task is to write clean, functional, idiomatic, and well-structured code in {language} to implement the features and architecture described in the following design documents.

    Based *strictly* on the design documents provided below, generate the complete source code for the application or the relevant modules. Ensure that the code includes:

    - Necessary imports and dependencies for {language}.
    - Clear function and class definitions that align with the components and modules outlined in the Technical Design.
    - Implementation of the key features and user flows described in the Functional Design.
    - Basic error handling where appropriate for common scenarios (e.g., invalid input).
    - Comments to explain key logic and complex sections, following {language} conventions.
    - Adherence to common {language} coding standards and best practices.
    - Consideration of the data models defined in the Technical Design.
    - Implementation of any API endpoints or database interactions described in the Technical Design (at a basic level).

    **Design Documents:**
    {state['design_docs']}

    **User Stories (for context):**
    {state['user_stories']}

    Return ONLY the raw {language} code, without any introductory text, explanations, or markdown fences. Ensure the code is ready to be saved into appropriate files and is syntactically correct.
    """

    # prompt = f"""
    # You are an expert software developer specializing in {language}. Your task is to write clean, functional, idiomatic, and well-structured code in {language} to implement the features and architecture described in the following design documents.

    # Based *strictly* on the design documents provided below, generate the complete source code for the application or the relevant modules. Ensure that the code includes:

    # - Necessary imports and dependencies for {language}.
    # - Clear function and class definitions that align with the components and modules outlined in the Technical Design.
    # - Implementation of the key features and user flows described in the Functional Design.
    # - Basic error handling where appropriate for common scenarios (e.g., invalid input).
    # - Comments to explain key logic and complex sections, following {language} conventions.
    # - Adherence to common {language} coding standards and best practices.
    # - Consideration of the data models defined in the Technical Design.
    # - Implementation of any API endpoints or database interactions described in the Technical Design (at a basic level).

    # **Design Documents:**
    # {state['design_docs']}

    # **User Stories (for context):**
    # {state['user_stories']}

    # Return the complete, raw {language} code, followed by a concise explanation of the key design decisions and how the generated code aligns with the provided Design Documents.

    # The explanation should be brief (aim for 2-3 sentences highlighting the main components, features implemented, and how they map to the design). Focus on the overall structure and significant implementation choices.

    # The format of your response should be:

    # ```
    # [Complete, raw {language} code]
    # Explanation: [Concise explanation of the generated code and its alignment with the Design Documents.]
    # ```

    # Ensure the code is ready to be saved into appropriate files and is syntactically correct. Comments within the code should be used for detailed logic explanations, while the separate explanation should provide a high-level overview.
    # """
    with st.spinner(f"Generating {language} Code..."):
        if llm:
            result = llm.invoke([HumanMessage(content=prompt)])
            content = clean_llm_code_output(result.content, language)
            state["code"] = content
            state["stage"] = "Code Review"
            state["history"].append(("generate_code", content)) # Keep history
            return state
        else:
            st.warning("LLM object is not initialized. Cannot generate code.")
            return state

# def code_review(state: SDLCState):
def code_review(state: SDLCState):
    global llm
    language = state.get('target_language', 'the specified language') # Default text if missing
    code_lang_tag = language.lower() if language else ''
    # prompt = f"""
    # You are an expert Code Reviewer for {language} code. Your task is to meticulously review the following {language} code, ensuring it aligns with the provided design documents and adheres to best practices for the language.

    # Review the code for the following:

    # - **Quality and Readability:** Is the code well-formatted, easy to understand, and consistently styled? Are variable and function names descriptive?
    # - **Adherence to {language} Standards and Idioms:** Does the code follow common {language} coding conventions and idioms (e.g., PEP 8 for Python, coding style guides for other languages)?
    # - **Correctness and Potential Bugs:** Does the code appear to implement the intended functionality as described in the design documents? Are there any obvious logical errors, potential runtime exceptions, or edge cases that are not handled?
    # - **Efficiency (Briefly):** Are there any immediately apparent inefficiencies in terms of performance or resource usage?
    # - **Security (Basic Awareness):** Are there any obvious basic security vulnerabilities (e.g., hardcoded credentials, lack of input sanitization - if applicable to the code's context)?
    # - **Completeness:** Does the code seem to cover all the key aspects outlined in the design documents?
    # - **Error Handling:** Is there appropriate error handling for potential issues?

    # **Respond in one of two ways:**

    # 1. **Approval:** If the code is well-written, adheres to standards, appears correct, and adequately implements the design, respond with ONLY:
    #    `approved`

    # 2. **Feedback:** If there are issues or areas for improvement, respond with ONLY:
    #    `feedback: [provide specific and actionable suggestions for fixing them in {language}. For each issue, clearly indicate the line number or section of code if possible, and explain the problem and how to correct it. Be concise but provide enough detail for the developer to understand the necessary changes. Focus on code quality, correctness, adherence to standards, and alignment with the design.]`

    # **Design Documents:**
    # {state['design_docs']}

    # **Code ({language}):**
    # ```
    # {code_lang_tag}
    # {state['code']}
    # ```
    # """

    prompt = f"""
    You are an expert Code Reviewer for {language} code. Your task is to meticulously review the following {language} code, ensuring it aligns with the provided design documents and adheres to best practices for the language.

    Review the code for the following:

    - **Quality and Readability:** Is the code well-formatted, easy to understand, and consistently styled? Are variable and function names descriptive?
    - **Adherence to {language} Standards and Idioms:** Does the code follow common {language} coding conventions and idioms (e.g., PEP 8 for Python, coding style guides for other languages)?
    - **Correctness and Potential Bugs:** Does the code appear to implement the intended functionality as described in the design documents? Are there any obvious logical errors, potential runtime exceptions, or edge cases that are not handled?
    - **Efficiency (Briefly):** Are there any immediately apparent inefficiencies in terms of performance or resource usage?
    - **Security (Basic Awareness):** Are there any obvious basic security vulnerabilities (e.g., hardcoded credentials, lack of input sanitization - if applicable to the code's context)?
    - **Completeness:** Does the code seem to cover all the key aspects outlined in the design documents?
    - **Error Handling:** Is there appropriate error handling for potential issues?

    **Respond in one of two ways:**

    1. **Approval with Explanation:** If the code is well-written, adheres to standards, appears correct, and adequately implements the design with no significant issues found, respond with:
        `approved: [provide a brief, concise explanation highlighting the code's strengths, such as good readability, clear structure, or strong adherence to design.]`

    2. **Feedback with Explanation:** If there are issues or areas for improvement, respond with:
        `feedback: [Provide a concise list of actionable suggestions for fixing them in {language}. For each issue, use a bullet point format:]
        - **Location:** [Indicate the line number or section of code if possible.]
        - **Problem:** [Concisely explain the issue (e.g., potential bug, style violation, missing error handling).]
        - **Suggestion/Explanation:** [Provide a clear, actionable suggestion on how to correct it, explaining why the change is needed. Keep this explanation brief.]
        [Focus your feedback on code quality, correctness, adherence to standards, and alignment with the design.]`

    **Design Documents:**
    {state['design_docs']}

    **Code ({language}):**
    ```
    {code_lang_tag}
    {state['code']}
    ```
    """

    with st.spinner(f"AI Reviewing {language} Code..."):
        if llm:
            result = llm.invoke([HumanMessage(content=prompt)])
            content = result.content.strip().lower()
            if content.startswith("approved"):
                state["decision"] = "approved"
                state["feedback"] = None
            elif content.startswith("feedback:"):
                state["decision"] = "feedback"
                state["feedback"] = content.split("feedback:", 1)[1].strip()
            else:
                st.warning(f"Code Review LLM sent unexpected response: '{content}'. Assuming feedback is needed.")
                state["decision"] = "feedback"
                state["feedback"] = f"LLM response unclear ({language} Code Review), please manually review and revise: '{content}'"
            state["history"].append(("code_review", result.content)) # Keep history
            return state
        else:
            st.warning("LLM object is not initialized. Cannot review code.")
            return state


# def fix_code_review(state: SDLCState):
def fix_code_review(state: SDLCState):
    global llm
    language = state.get('target_language', 'Python') # Default ok here as fallback
    code_lang_tag = language.lower()
    prompt = f"""
    You are an expert software developer tasked with fixing the following {language} code based *only* on the feedback provided. Your goal is to address the specific issues mentioned in the feedback while ensuring the corrected code remains functional, readable, idiomatic for {language}, and aligns with the original design.

    **Feedback:**
    {state.get('feedback', 'No feedback provided.')}

    **Original {language} Code:**
    ```
    {code_lang_tag}
    {state['code']}
    ```

    **Design Documents (for context):**
    {state['design_docs']}

    Return ONLY the complete, fixed raw {language} code, without any surrounding explanation, markdown fences, or comments unless they are part of the idiomatic coding style for {language} within the code itself. Ensure all necessary imports and the overall structure of the original code are maintained.
    """

    # prompt = f"""
    # You are an expert software developer tasked with fixing the following {language} code based *only* on the feedback provided. Your goal is to address the specific issues mentioned in the feedback while ensuring the corrected code remains functional, readable, idiomatic for {language}, and aligns with the original design.

    # **Feedback:**
    # {state.get('feedback', 'No feedback provided.')}

    # **Original {language} Code:**
    # ```
    # {code_lang_tag}
    # {state['code']}
    # ```

    # **Design Documents (for context):**
    # {state['design_docs']}

    # Return the complete, fixed raw {language} code, followed by a concise explanation of the changes you made and why they were necessary to address the feedback.

    # The explanation should be brief (1-2 sentences per feedback point, if applicable) and focus on how the changes directly address the issues raised in the feedback.

    # The format of your response should be:

    # ```
    # [Complete, fixed raw {language} code]
    # Explanation: [Concise explanation of changes, referencing the feedback.]
    # ```

    # Ensure all necessary imports and the overall structure of the original code are maintained in the fixed version. Comments within the code should only be included if they are part of the idiomatic coding style for {language}.
    # """

    with st.spinner(f"Fixing {language} Code based on review feedback..."):
        if llm and state.get('feedback'):
            result = llm.invoke([HumanMessage(content=prompt)])
            content = clean_llm_code_output(result.content, language)
            state["code"] = content
            state["decision"] = None
            state["feedback"] = None
            state["stage"] = "Code Review" # Go back to review the fix
            state["history"].append(("fix_code_review", content)) # Keep history   
            return state
        elif not state.get('feedback'):
            st.info("No feedback provided. Skipping code fixing.")
            state["stage"] = "Code Review" # Still go back to review
            state["history"].append(("fix_code_review", "No feedback provided. Skipping code fixing."))
            return state
        else:
            st.warning("LLM object is not initialized. Cannot fix code.")
            return state
        



def security_review(state: SDLCState):
    global llm
    language = state.get('target_language', 'the target language')
    code = state.get('code', '')
    code_lang_tag = language.lower() if language else ''
    design_docs = state.get('design_docs', '')

    if not code:
        st.warning("Cannot perform security review as code is missing.")
        state["decision"] = "feedback"
        state["feedback"] = "No code available for security review."
        return state

    # More comprehensive and context-aware security concerns
    security_concerns = ""
    if 'python' in language.lower():
        security_concerns = "potential for injection vulnerabilities (SQL, command, etc., especially if interacting with external systems or user input), insecure use of libraries (e.g., pickle, eval), improper handling of sensitive data, hardcoded secrets, cross-site scripting (XSS) if web-related, and denial-of-service (DoS) possibilities."
    elif 'java' in language.lower():
        security_concerns = "SQL injection, cross-site scripting (XSS), insecure deserialization, improper error handling leading to information disclosure, vulnerabilities in third-party libraries, and insufficient input validation."
    elif 'javascript' in language.lower():
        security_concerns = "cross-site scripting (XSS), prototype pollution, insecure handling of user input, vulnerabilities in frontend frameworks and libraries, and improper authentication/authorization mechanisms."
    elif 'c#' in language.lower():
        security_concerns = "SQL injection, cross-site scripting (XSS), insecure deserialization, buffer overflows (if using unsafe code), and improper handling of exceptions."
    elif 'go' in language.lower():
        security_concerns = "SQL injection, command injection, cross-site scripting (XSS) if web-related, improper handling of errors, and vulnerabilities in external packages."
    else:
        security_concerns = "common web application vulnerabilities such as cross-site scripting (XSS) and insecure handling of user input, as well as language-specific security best practices."

    # prompt = f"""
    # You are a highly skilled Security Analyst performing a security review of the following {language} code. Your task is to identify potential security vulnerabilities based on common attack vectors and best practices for secure coding in {language}.

    # Specifically, review the code for:

    # - **Input Validation:** Is user input properly validated and sanitized to prevent injection attacks (e.g., SQL injection, command injection, XSS)?
    # - **Authentication and Authorization:** Are there any obvious flaws in authentication or authorization mechanisms? (e.g., hardcoded credentials, insecure session management - if applicable).
    # - **Data Handling:** Is sensitive data (if any) handled securely? Are there any signs of hardcoded secrets or insecure storage?
    # - **Error Handling:** Does the error handling expose sensitive information?
    # - **Use of Libraries:** Are there any uses of known insecure libraries or functions?
    # - **Cross-Site Scripting (XSS):** If the code involves web output, are there potential XSS vulnerabilities?
    # - **Other Common Vulnerabilities:** Based on your knowledge of {language} security, look for other common pitfalls such as {security_concerns}.

    # Consider the design documents below to understand the intended functionality and identify potential discrepancies or missing security considerations.

    # **Design Documents:**
    # {design_docs}

    # **Code ({language}):**
    # ```
    # {code_lang_tag}
    # {code}
    # ```

    # Respond ONLY with 'approved' if no obvious high-risk security vulnerabilities are found based on a static analysis of the code.

    # Respond ONLY with 'feedback: [concise list of potential security vulnerabilities found in the code, specific to {language}. For each vulnerability, briefly explain the potential impact and suggest how to mitigate it. If possible, indicate the relevant code section or line number.]'
    # """

    prompt = f"""
    You are a highly skilled Security Analyst with extensive experience in performing security reviews of {language} code. Your primary task is to identify potential security vulnerabilities in the following code based on well-known attack vectors and established best practices for secure coding in {language}.

    Your analysis should specifically focus on, but not be limited to:

    - **Input Validation and Sanitization:** Scrutinize how user input is handled. Are there any instances where input is not properly validated or sanitized, potentially leading to injection attacks such as SQL injection, command injection, or Cross-Site Scripting (XSS)? Explain the potential attack vector and suggest appropriate sanitization or validation techniques in {language}.

    - **Authentication and Authorization Flaws:** Examine the authentication and authorization mechanisms (if present). Are there any obvious weaknesses like hardcoded credentials, insecure session management, or insufficient access controls? If so, describe the potential impact and recommend secure alternatives.

    - **Sensitive Data Handling:** Analyze how sensitive data is managed. Are there any indications of hardcoded secrets, insecure storage of credentials or personal information, or potential for data leaks? Detail the risks and suggest secure storage or handling methods.

    - **Error Handling Vulnerabilities:** Review the error handling mechanisms. Do error messages reveal sensitive information that could be exploited by attackers? Explain the potential information disclosure and suggest safer error handling practices.

    - **Insecure Use of Libraries and Functions:** Identify any usage of known vulnerable libraries, deprecated functions, or insecure coding patterns that are common in {language}. Explain the associated risks and recommend safer alternatives or updated libraries.

    - **Cross-Site Scripting (XSS) Potential:** If the code generates web output, meticulously look for potential XSS vulnerabilities. Explain how an attacker might inject malicious scripts and suggest appropriate output encoding or sanitization techniques in {language} to prevent XSS.

    - **Other {language}-Specific Security Concerns:** Based on your deep understanding of {language} security, identify any other common vulnerabilities relevant to this language, such as {security_concerns}. Explain the vulnerability and suggest mitigation strategies.

    Consider the provided design documents to understand the intended functionality of the code. Highlight any potential security vulnerabilities that seem to contradict the design or indicate missing security considerations.

    **Design Documents:**
    {design_docs}

    **Code ({language}):**
    ```
    {code_lang_tag}
    {code}
    ```

    You must respond in one of two formats:

    **Format 1: Approved**
    Respond **ONLY** with the word 'approved' if, after a thorough static analysis, you find no obvious high-risk security vulnerabilities.

    **Format 2: Feedback with Explanations and Mitigation**
    Respond **ONLY** with the phrase 'feedback:' followed by a well-structured, concise list of potential security vulnerabilities found in the code. For each identified vulnerability, please provide the following details:

    - **Vulnerability:** A clear and descriptive name of the security vulnerability (e.g., SQL Injection, XSS, Hardcoded Password).
    - **Location:** Specify the relevant code section or line number(s) where the vulnerability is likely to exist.
    - **Potential Impact:** Briefly explain the potential consequences and risks associated with this vulnerability.
    - **Mitigation:** Provide a specific and actionable recommendation on how to fix or mitigate this vulnerability in the context of {language} code. Include code examples if appropriate.

    Ensure your feedback is specific to {language} and directly addresses the identified vulnerabilities in the provided code.
    """

    with st.spinner(f"AI Performing Security Scan for {language}..."):
        if llm:
            result = llm.invoke([HumanMessage(content=prompt)])
            content = result.content.strip().lower()
            if content.startswith("approved"):
                state["decision"] = "approved"
                state["feedback"] = None
            elif content.startswith("feedback:"):
                state["decision"] = "feedback"
                state["feedback"] = content.split("feedback:", 1)[1].strip()
            else: # Assume feedback if not explicitly approved
                state["decision"] = "feedback"
                state["feedback"] = f"Potential security concerns identified ({language}): {content}"
            state["stage"] = "Security Review"
            state["history"].append(("security_review", result.content)) # Keep history
            return state
        else:
            st.warning("LLM object is not initialized. Cannot perform security review.")
            state["decision"] = "feedback"
            state["feedback"] = "LLM not initialized, skipping security review."
            state["stage"] = "Security Review"
            return state



def fix_security_issues(state: SDLCState):
    global llm
    language = state.get('target_language', 'Python')
    code = state.get('code', '')
    code_lang_tag = language.lower()
    feedback = state.get('feedback')
    design_docs = state.get('design_docs', '')

    if not feedback:
        st.info("No security feedback provided. Skipping code fixing.")
        state["stage"] = "Security Review" # Still go back for re-review
        return state

    prompt = f"""
    You are a highly skilled and security-conscious software developer. Your task is to fix the following {language} code based *only* on the security feedback provided below. It is crucial that you prioritize addressing the mentioned vulnerabilities securely and according to best practices for {language}.

    **Security Feedback:**
    {feedback}

    **Original {language} Code:**
    ```
    {code_lang_tag}
    {code}
    ```

    **Design Documents (for context):**
    {design_docs}

    Return ONLY the complete, fixed raw {language} code, without any surrounding explanation, markdown fences, or comments unless they are essential for the fix and follow idiomatic {language} commenting style. Ensure the corrected code is secure, functional, and still adheres to the original design where applicable.
    """

    # prompt = f"""
    # You are a highly skilled and security-conscious software developer. Your task is to fix the following {language} code based *only* on the security feedback provided below. It is crucial that you prioritize addressing the mentioned vulnerabilities securely and according to best practices for {language}.

    # **Security Feedback:**
    # {feedback}

    # **Original {language} Code:**
    # ```
    # {code_lang_tag}
    # {code}
    # ```

    # **Design Documents (for context):**
    # {design_docs}

    # Return the complete, fixed raw {language} code, followed by a concise explanation of the security fixes you implemented to address the feedback.

    # The explanation should briefly describe each vulnerability identified in the feedback and clearly state how your changes in the code mitigate that specific vulnerability. Aim for 1-2 sentences per feedback point.

    # The format of your response should be:

    # ```
    # [Complete, fixed raw {language} code]
    # Explanation of Security Fixes: [Concise explanation of each fix, referencing the security feedback.]
    # ```

    # Ensure the corrected code is secure, functional, and still adheres to the original design where applicable. Comments within the code should only be included if they are essential for the fix and follow idiomatic {language} commenting style.
    # """
    with st.spinner(f"Fixing {language} Code based on security feedback..."):
        if llm:
            result = llm.invoke([HumanMessage(content=prompt)])
            content = clean_llm_code_output(result.content, language)
            state["code"] = content
            state["decision"] = None
            state["feedback"] = None
            state["stage"] = "Security Review" # Go back for re-review
            state["history"].append(("fix_security_issues", content)) # Keep history
            return state
        else:
            st.warning("LLM object is not initialized. Cannot fix security issues.")
            return state




def write_test_cases(state: SDLCState):
    global llm
    language = state.get('target_language', 'Python')
    code_lang_tag = language.lower()
    # Suggest appropriate test framework based on language
    framework_suggestion = ""
    if language.lower() == 'python':
        framework_suggestion = "using `unittest` or `pytest`"
    elif language.lower() == 'java':
        framework_suggestion = "using JUnit 5"
    elif language.lower() == 'javascript':
        framework_suggestion = "using Jest or Mocha/Chai"
    elif language.lower() == 'go':
        framework_suggestion = "using the standard `testing` package"
    elif language.lower() == 'c#':
        framework_suggestion = "using MSTest or NUnit"
    else:
        framework_suggestion = f"using a common testing framework for {language}"

    prompt = f"""
    You are a highly skilled software quality assurance engineer. Your task is to write comprehensive and idiomatic {language} test cases using the {framework_suggestion} framework for the provided code.

    Your goal is to ensure the correctness and robustness of the code by covering a wide range of scenarios. For each public function or method in the code, write a set of unit tests that address the following:

    1. **Positive Tests (Happy Path):** Verify the expected behavior of the function with valid inputs.
    2. **Negative Tests:** Test how the function handles invalid, unexpected, or malformed inputs. Consider different types of invalidity (e.g., incorrect data types, out-of-range values, missing parameters).
    3. **Edge Cases:** Explore boundary conditions and less common scenarios that might reveal issues. This includes testing with empty inputs, very large inputs, zero values, null values (if applicable), and inputs at the limits of acceptable ranges.
    4. **Error Handling:** If the code is expected to raise specific exceptions or return error codes, write tests to ensure this behavior is correct.
    5. **Basic Functionality:** Ensure all core functionalities of each public method are tested.
    6. **State Changes (if applicable):** If the function modifies the state of an object or the system, write tests to verify these state changes.

    For each test case, please ensure it is:
    - **Independent:** Each test should be able to run in isolation without relying on the outcome of other tests.
    - **Clear and Readable:** The test code should be easy to understand and maintain. Use descriptive test names.
    - **Assertive:** Each test should include clear assertions to verify the expected outcome. Use appropriate assertion methods from the {framework_suggestion} framework.
    - **Well-Structured:** Organize your tests logically, potentially using test classes or suites if the framework supports it.
    - **Include necessary setup (e.g., instantiating classes, defining test data) and imports for {language}.**

    Code ({language}):
    ```
    {state['code']}
    ```

    Return ONLY the raw {language} test code, without any surrounding explanation, markdown fences, or comments unless they are part of the idiomatic testing style for {language} within the test functions themselves.
    """

    # prompt = f"""
    # You are a highly skilled software quality assurance engineer. Your task is to write comprehensive and idiomatic {language} test cases using the {framework_suggestion} framework for the provided code.

    # Your goal is to ensure the correctness and robustness of the code by covering a wide range of scenarios. For each public function or method in the code, write a set of unit tests that address the following:

    # 1. **Positive Tests (Happy Path):** Verify the expected behavior of the function with valid inputs.
    # 2. **Negative Tests:** Test how the function handles invalid, unexpected, or malformed inputs. Consider different types of invalidity (e.g., incorrect data types, out-of-range values, missing parameters).
    # 3. **Edge Cases:** Explore boundary conditions and less common scenarios that might reveal issues. This includes testing with empty inputs, very large inputs, zero values, null values (if applicable), and inputs at the limits of acceptable ranges.
    # 4. **Error Handling:** If the code is expected to raise specific exceptions or return error codes, write tests to ensure this behavior is correct.
    # 5. **Basic Functionality:** Ensure all core functionalities of each public method are tested.
    # 6. **State Changes (if applicable):** If the function modifies the state of an object or the system, write tests to verify these state changes.

    # For each test case, please ensure it is:
    # - **Independent:** Each test should be able to run in isolation without relying on the outcome of other tests.
    # - **Clear and Readable:** The test code should be easy to understand and maintain. Use descriptive test names.
    # - **Assertive:** Each test should include clear assertions to verify the expected outcome. Use appropriate assertion methods from the {framework_suggestion} framework.
    # - **Well-Structured:** Organize your tests logically, potentially using test classes or suites if the framework supports it.
    # - **Include necessary setup (e.g., instantiating classes, defining test data) and imports for {language}.**

    # Code ({language}):
    # ```
    # {state['code']}
    # ```

    # Return the complete, raw {language} test code, followed by a concise explanation of the testing strategy you employed.

    # The explanation should briefly outline how the generated test cases cover the different categories mentioned above (positive, negative, edge cases, error handling, etc.) and how they aim to ensure the robustness and correctness of the provided code. Aim for 2-3 sentences highlighting the main testing approaches used.

    # The format of your response should be:

    # ```
    # [Complete, raw {language} test code]
    # Explanation of Testing Strategy: [Concise explanation of how the tests cover different scenarios.]
    # ```

    # Comments within the test code should be used to explain the purpose of individual tests, following idiomatic {language} testing style. The separate explanation should provide a high-level overview of the testing approach.
    # """

    with st.spinner(f"Writing {language} Test Cases..."):
        if llm:
            result = llm.invoke([HumanMessage(content=prompt)])
            content = clean_llm_code_output(result.content, language)
            state["test_cases"] = content
            state["stage"] = "Test Case Review"
            state["history"].append(("write_test_cases", content)) # Keep history
            return state
        else:
            st.warning("LLM object is not initialized. Cannot write test cases.")
            return state



def review_test_cases(state: SDLCState):
    global llm
    language = state.get('target_language', 'the specified language')
    code_lang_tag = language.lower() if language else ''

    # Infer the testing framework from the language (consistent with write_test_cases)
    framework_suggestion = ""
    if language.lower() == 'python':
        framework_suggestion = "unittest or pytest"
    elif language.lower() == 'java':
        framework_suggestion = "JUnit 5"
    elif language.lower() == 'javascript':
        framework_suggestion = "Jest or Mocha/Chai"
    elif language.lower() == 'go':
        framework_suggestion = "the standard `testing` package"
    elif language.lower() == 'c#':
        framework_suggestion = "MSTest or NUnit"
    else:
        framework_suggestion = f"a common testing framework for {language}"

    # prompt = f"""
    # You are a highly skilled software quality assurance engineer tasked with reviewing {language} test cases written using the {framework_suggestion} framework. Your goal is to assess the quality and completeness of these tests based on the likely functionality of the provided code.

    # Consider the following criteria during your review:

    # - **Clarity and Readability:** Are the test names descriptive? Is the test code easy to understand? Is the setup and assertion logic clear?
    # - **Relevance to Code Functionality:** Do the tests seem to target the key public functionalities of the code (which you can infer from the test names and structure)?
    # - **Test Coverage:**
    #     - **Happy Path:** Are there tests covering the expected behavior with valid inputs?
    #     - **Negative Tests:** Are there tests that attempt to use invalid, unexpected, or malformed inputs?
    #     - **Edge Cases:** Are there tests for boundary conditions and less common scenarios (e.g., empty inputs, large inputs, zero values, null values)?
    #     - **Error Handling:** If the code likely involves error handling (e.g., raising exceptions), are there tests to verify this?
    # - **Correct Use of Framework:** Are the tests using the conventions and assertion methods of the {framework_suggestion} framework appropriately for {language}?
    # - **Completeness:** Based on the likely functionality, are there any obvious missing test scenarios?

    # Respond in one of two ways:

    # 1. **Approval:** If the tests appear reasonable and cover the essential aspects, respond with ONLY:
    #    `approved`

    # 2. **Feedback:** If there are areas for improvement, respond with ONLY:
    #    `feedback: [concise, actionable suggestions for improving the {language} tests. Be specific and provide examples where possible. Focus on missing scenarios, unclear assertions, non-idiomatic code, or incorrect framework usage.]`

    # Code (inferred from the context of these test cases):
    # ```
    # {state.get('code_language_tag', code_lang_tag)}
    # {state['code']}
    # ```

    # Test Cases ({language}):
    # ```
    # {code_lang_tag}
    # {state['test_cases']}
    # ```
    # """

    prompt = f"""
    You are a highly skilled software quality assurance engineer tasked with reviewing {language} test cases written using the {framework_suggestion} framework. Your goal is to assess the quality and completeness of these tests based on the likely functionality of the provided code.

    Consider the following criteria during your review:

    - **Clarity and Readability:** Are the test names descriptive? Is the test code easy to understand? Is the setup and assertion logic clear?
    - **Relevance to Code Functionality:** Do the tests seem to target the key public functionalities of the code (which you can infer from the test names and structure)?
    - **Test Coverage:**
        - **Happy Path:** Are there tests covering the expected behavior with valid inputs?
        - **Negative Tests:** Are there tests that attempt to use invalid, unexpected, or malformed inputs?
        - **Edge Cases:** Are there tests for boundary conditions and less common scenarios (e.g., empty inputs, large inputs, zero values, null values)?
        - **Error Handling:** If the code likely involves error handling (e.g., raising exceptions), are there tests to verify this?
    - **Correct Use of Framework:** Are the tests using the conventions and assertion methods of the {framework_suggestion} framework appropriately for {language}?
    - **Completeness:** Based on the likely functionality, are there any obvious missing test scenarios?

    Respond in one of two ways:

    1. **Approval with Explanation:** If the tests appear reasonable and cover the essential aspects, respond with:
        `approved: [briefly explain why the tests are considered good, highlighting 1-2 key strengths like good coverage of happy path or clear test structure.]`

    2. **Feedback with Explanation:** If there are areas for improvement, respond with:
        `feedback: [concise, actionable suggestions for improving the {language} tests. Be specific and provide examples where possible. Focus on missing scenarios, unclear assertions, non-idiomatic code, or incorrect framework usage. Briefly explain the reasoning behind each suggestion.]`

    Code (inferred from the context of these test cases):
    ```
    {state.get('code_language_tag', code_lang_tag)}
    {state['code']}
    ```

    Test Cases ({language}):
    ```
    {code_lang_tag}
    {state['test_cases']}
    ```
    """
    with st.spinner(f"AI Reviewing {language} Test Cases..."):
        if llm:
            result = llm.invoke([HumanMessage(content=prompt)])
            content = result.content.strip().lower()
            if content == "approved":
                state["decision"] = "approved"
                state["feedback"] = None
            elif content.startswith("feedback:"):
                state["decision"] = "feedback"
                state["feedback"] = content.split("feedback:", 1)[1].strip()
            else:
                st.warning(f"Test Review LLM sent unexpected response: '{content}'. Assuming feedback is needed.")
                state["decision"] = "feedback"
                state["feedback"] = f"LLM response unclear ({language} Test Review), please manually review and revise: '{content}'"
            state["history"].append(("review_test_cases", result.content)) # Keep history
            return state
        else:
            st.warning("LLM object is not initialized. Cannot review test cases.")
            return state



def fix_test_cases(state: SDLCState):
    global llm
    language = state.get('target_language', 'Python')
    code_lang_tag = language.lower()

    # Infer the testing framework (consistent with write_test_cases)
    framework_suggestion = ""
    if language.lower() == 'python':
        framework_suggestion = "unittest or pytest"
    elif language.lower() == 'java':
        framework_suggestion = "JUnit 5"
    elif language.lower() == 'javascript':
        framework_suggestion = "Jest or Mocha/Chai"
    elif language.lower() == 'go':
        framework_suggestion = "the standard `testing` package"
    elif language.lower() == 'c#':
        framework_suggestion = "MSTest or NUnit"
    else:
        framework_suggestion = f"a common testing framework for {language}"

    prompt = f"""
    You are a highly skilled software quality assurance engineer. Your task is to update the existing {language} test cases, written using the {framework_suggestion} framework, based *only* on the feedback provided below.

    Your goal is to address the specific issues raised in the feedback while ensuring the updated tests remain comprehensive, idiomatic for {language}, and adhere to the principles of good unit testing.

    **Feedback:**
    {state.get('feedback', 'No feedback provided.')}

    **Original {language} Test Cases:**
    ```
    {code_lang_tag}
    {state['test_cases']}
    ```

    **Code Under Test (for context):**
    ```
    {state.get('code_language_tag', code_lang_tag)}
    {state['code']}
    ```

    Return ONLY the complete, updated raw {language} test code, without any surrounding explanation, markdown fences, or comments unless they are part of the idiomatic testing style for {language} within the test functions themselves. Ensure all necessary imports and setup are still present in the updated code.
    """

    # prompt = f"""
    # You are a highly skilled software quality assurance engineer. Your task is to update the existing {language} test cases, written using the {framework_suggestion} framework, based *only* on the feedback provided below.

    # Your goal is to address the specific issues raised in the feedback while ensuring the updated tests remain comprehensive, idiomatic for {language}, and adhere to the principles of good unit testing.

    # **Feedback:**
    # {state.get('feedback', 'No feedback provided.')}

    # **Original {language} Test Cases:**
    # ```
    # {code_lang_tag}
    # {state['test_cases']}
    # ```

    # **Code Under Test (for context):**
    # ```
    # {state.get('code_language_tag', code_lang_tag)}
    # {state['code']}
    # ```

    # Return the complete, updated raw {language} test code, followed by a concise explanation of the changes you made to address the feedback.

    # The explanation should briefly describe each point of feedback and clearly state how your modifications in the test code address that specific issue. Aim for 1-2 sentences per feedback point.

    # The format of your response should be:

    # ```
    # [Complete, updated raw {language} test code]
    # Explanation of Test Case Updates: [Concise explanation of each update, referencing the feedback.]
    # ```

    # Ensure all necessary imports and setup are still present in the updated code. Comments within the test code should only be included if they are part of the idiomatic testing style for {language} and explain the purpose of specific tests.
    # """
    with st.spinner(f"Fixing {language} Test Cases based on feedback..."):
        if llm and state.get('feedback'):
            result = llm.invoke([HumanMessage(content=prompt)])
            content = clean_llm_code_output(result.content, language)
            state["test_cases"] = content
            state["decision"] = None
            state["feedback"] = None
            state["stage"] = "Test Case Review" # Go back to review fixes
            state["history"].append(("fix_test_cases", content)) # Keep history
            return state
        elif not state.get('feedback'):
            st.info("No feedback provided. Skipping test case fixing.")
            state["stage"] = "Test Case Review" # Move to review even if no changes
            state["history"].append(("fix_test_cases", "No feedback provided. Skipping test case fixing."))
            return state
        else:
            st.warning("LLM object is not initialized. Cannot fix test cases.")
            return state




def qa_testing(state: SDLCState, show_llm_details: bool = False) -> SDLCState:
    """
    Performs quality assurance testing on the provided code and test cases using an LLM.

    The LLM acts as a software QA engineer, analyzing the code and tests to determine
    if the tests are likely to pass and adequately cover the code's functionality.

    Process of LLM-based QA Testing:
    1.  Prompt Generation: A detailed prompt is constructed that includes:
        -   The role of the LLM (a highly skilled software QA engineer).
        -   The task: evaluate code and test cases.
        -   Criteria: likelihood of tests passing and adequacy of test coverage.
        -   Context: the programming language of the code and tests.
        -   The code itself.
        -   The test cases.
        -   Required response format: either "PASS: [reasoning]" or "FAIL: [reasoning]".

    2.  LLM Invocation: The generated prompt is sent to the LLM. The LLM processes this prompt using its understanding of programming, testing principles, and common practices.

    3.  Analysis and Reasoning: The LLM analyzes the code to understand its functionality and the test cases to see what aspects of the code they are testing. It then performs a form of static analysis to:
        -   Assess the logical correctness of the code (to the extent possible without execution).
        -   Evaluate the comprehensiveness of the test cases. Does it seem like all major functionalities and edge cases are covered?
        -   Consider potential failure points in the code that might not be adequately addressed by the tests.
        -   Look for common testing best practices (e.g., clear assertions, test isolation).

    4.  Decision Making: Based on its analysis, the LLM makes a decision on whether the tests are likely to pass and provide adequate coverage.

    5.  Response Formatting: The LLM formats its response according to the specified format, providing a brief reasoning for its decision (either why it's likely to pass or why it might fail or is inadequate).

    6.  State Update: The function then parses the LLM's response:
        -   If the response starts with "PASS:", the state is updated with "decision": "passed" and the reasoning is stored in "feedback".
        -   If the response starts with "FAIL:", the state is updated with "decision": "failed" and the reasoning is stored in "feedback".
        -   If the response is in an unexpected format, the state is marked as "failed" and a message recommending manual review is stored in "feedback".

    7.  History Tracking: The prompt sent to the LLM and the raw content of the LLM's response are stored in the `state["history"]` list if `show_llm_details` is True, providing a detailed log of the LLM's interaction. The final processed output is also stored in the history.

    In essence, the LLM simulates a human QA engineer's thought process when reviewing code and tests, leveraging its broad knowledge base to identify potential issues and assess the quality of the testing strategy.
    """
    global llm
    language = state.get('target_language', 'the specified language')
    code = state.get('code', '')
    test_cases = state.get('test_cases', '')

    if not code or not test_cases:
        st.warning("Cannot perform QA testing as code or test cases are missing.")
        state["decision"] = "failed"
        state["feedback"] = "Code or test cases not found."
        state["stage"] = "QA Testing"
        return state



    prompt = f"""
    You are a highly skilled software quality assurance engineer. Your task is to evaluate the provided {language} code and its corresponding test cases to determine if the tests are likely to pass and if they provide adequate coverage of the code's functionality.

    Analyze the code to understand its intended behavior and then carefully review the test cases to see if they thoroughly test this behavior. Consider common programming practices, potential edge cases, and the overall structure of both the code and the tests.

    **Code ({language}):**
    ```
    {code}
    ```

    **Test Cases ({language}):**
    ```
    {test_cases}
    ```

    Based on your analysis, respond with one of the following:

    - `PASS: [Begin with a brief statement of why the tests are likely to pass. Then, concisely explain the key strengths in the test coverage, mentioning specific aspects of the code's functionality that are well-tested.]`

    - `FAIL: [Begin with a brief statement of why the tests are likely to fail or are inadequate. Then, concisely point out the significant weaknesses or gaps in the test coverage, ideally mentioning specific functionalities or edge cases that are missing or poorly tested.]`
    """

    with st.spinner(f"Simulating QA Testing for {language} code using AI analysis..."):
        if llm:
            state["history"] = state.get("history", [])
            if show_llm_details:
                st.subheader("Prompt sent to LLM for QA Testing:")
                # st.code(prompt, language='markdown')
                st.code(prompt, language='text')
                state["history"].append(("qa_testing_prompt", prompt))

            try:
                result = llm.invoke([HumanMessage(content=prompt)])
                content = result.content.strip()

                if show_llm_details:
                    st.subheader("Raw LLM Output for QA Testing:")
                    st.code(content, language='text')
                    state["history"].append(("qa_testing_raw_output", content))

                if content.startswith("PASS:"):
                    state["decision"] = "passed"
                    state["feedback"] = content.split("PASS:", 1)[1].strip()
                elif content.startswith("FAIL:"):
                    state["decision"] = "failed"
                    state["feedback"] = content.split("FAIL:", 1)[1].strip()
                else:
                    state["decision"] = "failed"
                    state["feedback"] = f"AI QA analysis returned an unexpected response: '{content}'. Manual review recommended."
                state["history"].append(("qa_testing_result", content)) # Keep history
                state["stage"] = "QA Testing"
            except Exception as e:
                st.error(f"An error occurred during LLM QA analysis: {e}")
                state["decision"] = "error"
                state["feedback"] = f"Error during AI QA analysis: {e}. Manual review required."
                state["history"].append(("qa_testing_error", str(e)))
                state["stage"] = "QA Testing"

        else:
            st.warning("LLM object is not initialized. Using basic simulation for QA Testing.")
            time.sleep(1) # Simulate work
            passed = True # Simulate passing QA for simplicity
            if passed:
                state["decision"] = "passed"
                state["feedback"] = "Basic QA simulation passed."
                state["history"].append(("qa_testing_simulation", "Basic QA simulation passed."))
            else:
                state["decision"] = "failed"
                state["feedback"] = f"Basic QA simulation failed for {language}. Check code or tests."
                state["history"].append(("qa_testing_simulation", f"Basic QA simulation failed for {language}. Check code or tests."))
            state["stage"] = "QA Testing"
    return state

def deploy(state: SDLCState, show_llm_details: bool = False) -> SDLCState:
    global llm

    language = state.get('target_language', '')
    qa_decision = state.get('decision', 'unknown')
    qa_feedback = state.get('feedback', 'No QA feedback available.')
    code = state.get('code', '')

    if qa_decision != "passed":
        st.warning("Deployment not recommended as QA testing did not pass.")
        state["stage"] = "Deployment Failed"
        state["feedback"] = "Deployment halted due to failed QA testing. Reason: " + qa_feedback
        return state

    prompt = f"""
    You are an experienced software deployment engineer. The software application written in {language} has successfully passed quality assurance testing. The QA outcome was: "{qa_decision.upper()}: {qa_feedback}".

    **Code ({language}):**
    ```
    {code}
    ```

    Considering the successful QA outcome, provide a brief explanation of why this suggests the deployment is likely to be successful. Highlight the positive implications of passing the QA stage for the deployment process.
    """

    with st.spinner(f"Simulating Deployment of {language} application with AI analysis..."):
        time.sleep(1) # Simulate deployment work

        if llm:
            if show_llm_details:
                st.subheader("Prompt sent to LLM for Deployment Reasoning:")
                st.code(prompt, language='markdown')

            try:
                result = llm.invoke([HumanMessage(content=prompt)])
                reasoning = result.content.strip()
                state["deployment_reasoning"] = reasoning

                if show_llm_details:
                    st.subheader("Raw LLM Output for Deployment Reasoning:")
                    st.code(reasoning, language='text')

                st.success(f"🚀 {language} Software Deployed Successfully!")
                st.info(f"Deployment Reasoning (AI): {reasoning}")
                state["history"] = state.get("history", []) + [("deploy", reasoning)] # Add to history
            except Exception as e:
                st.error(f"An error occurred during LLM deployment reasoning: {e}")
                state["deployment_reasoning"] = f"Error during AI deployment reasoning: {e}"
                st.success(f"🚀 {language} Software Deployed Successfully!") # Still mark as deployed for simulation purposes
                st.error("Error providing deployment reasoning. Manual review of logs recommended.")
                state["history"] = state.get("history", []) + [("deploy", f"Error during LLM reasoning: {e}")] # Add error to history

        else:
            st.warning("LLM object is not initialized. Using basic simulation for deployment reasoning.")
            state["deployment_reasoning"] = "Basic deployment simulation successful as QA passed."
            st.success(f"🚀 {language} Software Deployed Successfully!")
            st.info(f"Deployment Reasoning: Basic simulation successful as QA passed.")
            state["history"] = state.get("history", []) + [("deploy", "Basic deployment simulation successful as QA passed.")] # Add basic reasoning to history

        state["stage"] = "Deployed"
    return state

# -----------------------
#   streamlit app
# -----------------------

st.set_page_config(layout="wide", page_title="AI-Powered Multi-Language SDLC")
# st.title("🤖 AI-Powered Multi-Language SDLC Workflow")
st.title("🚀🔗 Automated Multi-Lingual DevStream")
st.caption("Using LangGraph concepts and Groq to simulate software development stages for various languages.")

# --- State Initialization (with target_language) ---
if 'app_state' not in st.session_state:
    st.session_state.app_state = SDLCState(
        stage="User Input",
        user_input="",
        target_language=None, # Initialize new field
        user_stories=None,
        design_docs=None,
        code=None,
        test_cases=None,
        decision=None,
        feedback=None,
        history=[]
    )
if 'feedback_input' not in st.session_state:
    st.session_state.feedback_input = ""
if 'show_feedback_box' not in st.session_state:
    st.session_state.show_feedback_box = False
if 'show_llm_details' not in st.session_state:
    st.session_state.show_llm_details = False

# Get current state
current_state = st.session_state.app_state


# --- Sidebar for API Key Input ---
# with st.sidebar:
#     st.header("⚙️ Configuration")
#     # --- API Key and Model Name Input ---
#     groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
#     groq_model_name = st.text_input("Enter the Groq Model Name (e.g., llama3-70b-8192):", value="llama3-70b-8192")

#     # Initialize LLM only if API key is provided
#     if groq_api_key:
#         llm = ChatGroq(model_name=groq_model_name, api_key=groq_api_key, temperature=0.1)
#     else:
#         st.warning("Please enter your Groq API key to start the process.")

# new added option in llms
# Define a list of common Groq models
# Note: This list might need updates as Groq introduces new models or retires old ones.

COMMON_GROQ_MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    # Add other models here as they become available and you wish to include them
]

with st.sidebar:
    st.header("⚙️ Configuration")
    # --- API Key Input ---
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")

    # --- Model Selection using Selectbox ---
    # Use st.selectbox to let the user choose from the list of common models
    groq_model_name = st.selectbox(
        "Select Groq Model:",
        COMMON_GROQ_MODELS,
        index=COMMON_GROQ_MODELS.index("llama3-70b-8192") if "llama3-70b-8192" in COMMON_GROQ_MODELS else 0 # Set a default value
    )


    # Initialize LLM only if API key is provided
    llm = None # Initialize llm to None by default
    if groq_api_key:
        try:
            # Initialize ChatGroq with the selected model and provided API key
            llm = ChatGroq(model_name=groq_model_name, api_key=groq_api_key, temperature=0.1)
            st.success(f"Groq LLM '{groq_model_name}' initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing Groq LLM: {e}")
            st.warning("Please check your API Key and Model Name.")
            llm = None # Ensure llm is None if initialization fails
    else:
        st.warning("Please enter your Groq API key to start the process.")
        llm = None # Ensure llm is None if API key is missing

# Now, the 'llm' variable is available for use in your main app logic
# You can check 'if llm:' before using it in your functions like qa_testing, deploy, etc.


    

    # st.header("LLM Output History")
    st.header("⚛️ Autonomous SDLC State History")
    with st.expander("Show Details", expanded=False):
        if current_state['history']:
            for item in current_state['history']:
                stage_name, output = item
                display_output(f"LLM Output - {stage_name}", output)
        else:
            st.info("No LLM output history yet.")



# --- Display Area ---
st.sidebar.header("Workflow Progress")
st.sidebar.info(f"Current Stage: **{current_state['stage']}**")
if current_state.get('target_language'):
    st.sidebar.write(f"Target Language: **{current_state['target_language']}**")


# Display artifacts generated so far (pass language to display_output where relevant)
if current_state['user_input']:
     with st.expander("Initial Requirements", expanded=False):
         st.markdown(f"**Requirements:**\n{current_state['user_input']}")
         if current_state.get('target_language'):
            st.markdown(f"**Target Language:** {current_state['target_language']}")

if current_state['user_stories']:
     with st.expander("User Stories", expanded=current_state['stage'] == "Product Owner Review"):
        # User stories are language agnostic, use default display
        display_output("User Stories", current_state['user_stories'])

if current_state['design_docs']:
     with st.expander("Design Documents", expanded=current_state['stage'] == "Design Review"):
        # Design docs are language agnostic, use default display
        display_output("Design Documents", current_state['design_docs'])

if current_state['code']:
     lang = current_state.get('target_language', 'code') # Get language for title/display
     with st.expander(f"Generated {lang.capitalize()} Code", expanded=current_state['stage'] in ["Code Review", "Security Review", "QA Testing"]):
         display_output(f"Generated {lang.capitalize()} Code", current_state['code'], language=lang) # Pass language

if current_state['test_cases']:
     lang = current_state.get('target_language', 'tests')
     with st.expander(f"{lang.capitalize()} Test Cases", expanded=current_state['stage'] in ["Test Case Review", "QA Testing"]):
         display_output(f"{lang.capitalize()} Test Cases", current_state['test_cases'], language=lang) # Pass language

st.divider()

# --- Main Interaction Logic ---

if current_state['stage'] == "User Input":
    st.header("1. Enter Requirements & Target Language")
    user_input_area = st.text_area("Describe the software you want to build:", height=150, key="user_input_main")
    target_lang_input = st.text_input("Target Programming Language (e.g., Python, Java, JavaScript, Go, C#):", key="target_lang_input")

    if st.button("Start SDLC Process", type="primary"):
        if user_input_area and target_lang_input:
            st.session_state.app_state['user_input'] = user_input_area
            # Store language consistently (e.g., lowercase)
            st.session_state.app_state['target_language'] = target_lang_input.strip().lower()
            st.session_state.app_state['stage'] = "Generate User Stories" # Set next stage
            st.rerun() # Rerun to process the next stage
        elif not user_input_area:
            st.warning("Please enter some requirements.")
        else:
            st.warning("Please enter the target programming language.")


elif current_state['stage'] == "Generate User Stories":
    st.session_state.app_state = generate_user_stories(current_state)
    st.rerun()


elif current_state['stage'] == "Product Owner Review":
    st.header("2. Product Owner Review (User Stories)")
    st.markdown("Review the generated user stories above.")
    # --- Identical logic as before ---
    if st.button("🤖 Ask AI to Review Stories"):
        st.session_state.app_state = product_owner_review(current_state)
        if st.session_state.app_state['decision'] == 'approved':
             st.session_state.app_state['stage'] = 'Create Design Docs'
        elif st.session_state.app_state['decision'] == 'feedback':
             st.session_state.app_state['stage'] = 'Revise User Stories'
             st.warning(f"AI Feedback: {st.session_state.app_state['feedback']}")
        st.rerun()

    st.markdown("--- OR ---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Manually", key="po_approve"):
            st.session_state.app_state['decision'] = 'approved'
            st.session_state.app_state['feedback'] = None
            st.session_state.app_state['stage'] = 'Create Design Docs'
            st.session_state.show_feedback_box = False
            st.rerun()
    with col2:
        if st.button("✍️ Provide Feedback Manually", key="po_feedback"):
            st.session_state.show_feedback_box = True

    if st.session_state.show_feedback_box:
        feedback_text = st.text_area("Enter your feedback for the user stories:", key="po_feedback_text", value=st.session_state.feedback_input)
        if st.button("Submit Feedback", key="po_submit_feedback"):
            if feedback_text:
                st.session_state.app_state['decision'] = 'feedback'
                st.session_state.app_state['feedback'] = feedback_text
                st.session_state.app_state['stage'] = 'Revise User Stories'
                st.session_state.show_feedback_box = False
                st.session_state.feedback_input = ""
                st.rerun()
            else:
                st.warning("Please enter feedback before submitting.")





elif current_state['stage'] == "Revise User Stories":
    st.header("Revising User Stories...")
    if current_state['feedback']:
         st.markdown(f"**Feedback Received:** {current_state['feedback']}")
    else:
         st.warning("No feedback found to revise.") # Safety check
    st.session_state.app_state = revise_user_stories(current_state)
    st.rerun()



elif current_state['stage'] == "Create Design Docs":
    st.header("3. Generating Design Documents...")
    st.session_state.app_state = create_design_docs(current_state)
    st.rerun()



elif current_state['stage'] == "Design Review":
    st.header("4. Design Document Review")
    st.markdown("Review the generated design documents above.")
    # --- Identical logic as before ---
    if st.button("🤖 Ask AI to Review Design"):
        st.session_state.app_state = design_review(current_state)
        if st.session_state.app_state['decision'] == 'approved':
             st.session_state.app_state['stage'] = 'Generate Code'
        elif st.session_state.app_state['decision'] == 'feedback':
             st.session_state.app_state['stage'] = 'Revise Design Docs'
             st.warning(f"AI Feedback: {st.session_state.app_state['feedback']}")
        st.rerun()

    st.markdown("--- OR ---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Manually", key="design_approve"):
            st.session_state.app_state['decision'] = 'approved'
            st.session_state.app_state['feedback'] = None
            st.session_state.app_state['stage'] = 'Generate Code'
            st.session_state.show_feedback_box = False
            st.rerun()
    with col2:
        if st.button("✍️ Provide Feedback Manually", key="design_feedback"):
            st.session_state.show_feedback_box = True

    if st.session_state.show_feedback_box:
        feedback_text = st.text_area("Enter your feedback for the design docs:", key="design_feedback_text", value=st.session_state.feedback_input)
        if st.button("Submit Feedback", key="design_submit_feedback"):
            if feedback_text:
                st.session_state.app_state['decision'] = 'feedback'
                st.session_state.app_state['feedback'] = feedback_text
                st.session_state.app_state['stage'] = 'Revise Design Docs'
                st.session_state.show_feedback_box = False
                st.session_state.feedback_input = ""
                st.rerun()
            else:
                st.warning("Please enter feedback before submitting.")


elif current_state['stage'] == "Revise Design Docs":
    st.header("Revising Design Docs...")
    if current_state['feedback']:
         st.markdown(f"**Feedback Received:** {current_state['feedback']}")
    else:
         st.warning("No feedback found to revise.")
    st.session_state.app_state = revise_design_docs(current_state)
    st.rerun()

elif current_state['stage'] == "Generate Code":
    st.header(f"5. Generating {current_state.get('target_language','Code').capitalize()} Code...")
    st.session_state.app_state = generate_code(current_state)
    # Check if generate_code changed stage back due to error (like missing lang)
    if st.session_state.app_state['stage'] != "Code Review":
         st.warning("Code generation skipped or failed, returning to previous step.")
    st.rerun()

elif current_state['stage'] == "Code Review":
    lang = current_state.get('target_language','Code').capitalize()
    st.header(f"6. {lang} Code Review")
    st.markdown(f"Review the generated {lang} code above.")
    # --- Updated logic structure for clarity ---
    ai_review_pressed = st.button(f"🤖 Ask AI for {lang} Code Review")
    st.markdown("--- OR ---")
    col1, col2 = st.columns(2)
    manual_approve_pressed = col1.button("✅ Approve Manually", key="code_approve")
    manual_feedback_pressed = col2.button("✍️ Provide Feedback Manually", key="code_feedback")

    if ai_review_pressed:
        st.session_state.app_state = code_review(current_state)
        if st.session_state.app_state['decision'] == 'approved':
             st.session_state.app_state['stage'] = 'Security Review'
        elif st.session_state.app_state['decision'] == 'feedback':
             st.session_state.app_state['stage'] = 'Fix Code Review'
             st.warning(f"AI Feedback: {st.session_state.app_state['feedback']}")
        st.rerun()

    if manual_approve_pressed:
        st.session_state.app_state['decision'] = 'approved'
        st.session_state.app_state['feedback'] = None
        st.session_state.app_state['stage'] = 'Security Review'
        st.session_state.show_feedback_box = False
        st.rerun()

    if manual_feedback_pressed:
        st.session_state.show_feedback_box = True
        # Rerun needed to show the box below if it wasn't already shown
        st.rerun()

    if st.session_state.show_feedback_box:
        feedback_text = st.text_area(f"Enter your feedback for the {lang} code:", key="code_feedback_text", value=st.session_state.feedback_input)
        if st.button("Submit Feedback", key="code_submit_feedback"):
            if feedback_text:
                st.session_state.app_state['decision'] = 'feedback'
                st.session_state.app_state['feedback'] = feedback_text
                st.session_state.app_state['stage'] = 'Fix Code Review'
                st.session_state.show_feedback_box = False
                st.session_state.feedback_input = ""
                st.rerun()
            else:
                st.warning("Please enter feedback before submitting.")


elif current_state['stage'] == "Fix Code Review":
    lang = current_state.get('target_language','Code').capitalize()
    st.header(f"Fixing {lang} Code based on Review...")
    if current_state['feedback']:
         st.markdown(f"**Feedback Received:** {current_state['feedback']}")
    else:
         st.warning("No feedback found to revise.")
    st.session_state.app_state = fix_code_review(current_state)
    st.rerun()

elif current_state['stage'] == "Security Review":
    lang = current_state.get('target_language','Code').capitalize()
    st.header(f"7. {lang} Security Review")
    st.markdown(f"Performing automated security check on the {lang} code above.")
    # --- Similar logic structure to Code Review ---
    ai_review_pressed = st.button(f"🤖 Ask AI for {lang} Security Review")
    st.markdown("--- OR ---")
    col1, col2 = st.columns(2)
    manual_approve_pressed = col1.button("✅ Approve Manually", key="sec_approve")
    manual_feedback_pressed = col2.button("✍️ Provide Security Feedback Manually", key="sec_feedback")

    if ai_review_pressed:
        st.session_state.app_state = security_review(current_state)
        if st.session_state.app_state['decision'] == 'approved':
             st.session_state.app_state['stage'] = 'Write Test Cases'
        elif st.session_state.app_state['decision'] == 'feedback':
             st.session_state.app_state['stage'] = 'Fix Security'
             st.warning(f"AI Security Feedback: {st.session_state.app_state['feedback']}")
        st.rerun()

    if manual_approve_pressed:
        st.session_state.app_state['decision'] = 'approved'
        st.session_state.app_state['feedback'] = None
        st.session_state.app_state['stage'] = 'Write Test Cases'
        st.session_state.show_feedback_box = False
        st.rerun()

    if manual_feedback_pressed:
        st.session_state.show_feedback_box = True
        st.rerun()

    if st.session_state.show_feedback_box:
        feedback_text = st.text_area(f"Enter security concerns for the {lang} code:", key="sec_feedback_text", value=st.session_state.feedback_input)
        if st.button("Submit Security Feedback", key="sec_submit_feedback"):
            if feedback_text:
                st.session_state.app_state['decision'] = 'feedback'
                st.session_state.app_state['feedback'] = feedback_text
                st.session_state.app_state['stage'] = 'Fix Security'
                st.session_state.show_feedback_box = False
                st.session_state.feedback_input = ""
                st.rerun()
            else:
                st.warning("Please enter feedback before submitting.")


elif current_state['stage'] == "Fix Security":
    lang = current_state.get('target_language','Code').capitalize()
    st.header(f"Fixing {lang} Code based on Security Review...")
    if current_state['feedback']:
         st.markdown(f"**Feedback Received:** {current_state['feedback']}")
    else:
         st.warning("No feedback found to revise.")
    st.session_state.app_state = fix_security_issues(current_state)
    st.rerun()

elif current_state['stage'] == "Write Test Cases":
    lang = current_state.get('target_language','Tests').capitalize()
    st.header(f"8. Writing {lang} Test Cases...")
    st.session_state.app_state = write_test_cases(current_state)
    st.rerun()

elif current_state['stage'] == "Test Case Review":
    lang = current_state.get('target_language','Test').capitalize()
    st.header(f"9. {lang} Case Review")
    st.markdown(f"Review the generated {lang} cases above.")
    # --- Similar logic structure to Code Review ---
    ai_review_pressed = st.button(f"🤖 Ask AI to Review {lang} Cases")
    st.markdown("--- OR ---")
    col1, col2 = st.columns(2)
    manual_approve_pressed = col1.button("✅ Approve Manually", key="test_approve")
    manual_feedback_pressed = col2.button(f"✍️ Provide {lang} Case Feedback Manually", key="test_feedback")

    if ai_review_pressed:
        st.session_state.app_state = review_test_cases(current_state)
        if st.session_state.app_state['decision'] == 'approved':
             st.session_state.app_state['stage'] = 'QA Testing'
        elif st.session_state.app_state['decision'] == 'feedback':
             st.session_state.app_state['stage'] = 'Fix Test Cases'
             st.warning(f"AI Test Case Feedback: {st.session_state.app_state['feedback']}")
        st.rerun()

    if manual_approve_pressed:
        st.session_state.app_state['decision'] = 'approved'
        st.session_state.app_state['feedback'] = None
        st.session_state.app_state['stage'] = 'QA Testing'
        st.session_state.show_feedback_box = False
        st.rerun()

    if manual_feedback_pressed:
        st.session_state.show_feedback_box = True
        st.rerun()

    if st.session_state.show_feedback_box:
        feedback_text = st.text_area(f"Enter feedback for {lang} cases:", key="test_feedback_text", value=st.session_state.feedback_input)
        if st.button(f"Submit {lang} Case Feedback", key="test_submit_feedback"):
            if feedback_text:
                st.session_state.app_state['decision'] = 'feedback'
                st.session_state.app_state['feedback'] = feedback_text
                st.session_state.app_state['stage'] = 'Fix Test Cases'
                st.session_state.show_feedback_box = False
                st.session_state.feedback_input = ""
                st.rerun()
            else:
                st.warning("Please enter feedback before submitting.")


elif current_state['stage'] == "Fix Test Cases":
    lang = current_state.get('target_language','Test').capitalize()
    st.header(f"Fixing {lang} Cases...")
    if current_state['feedback']:
         st.markdown(f"**Feedback Received:** {current_state['feedback']}")
    else:
         st.warning("No feedback found to revise.")
    st.session_state.app_state = fix_test_cases(current_state)
    st.rerun()


# elif current_state['stage'] == "QA Testing":
#     lang = current_state.get('target_language','').capitalize()
#     st.header(f"10. Simulating QA Testing ({lang})...")
#     # Call qa_testing with the show_llm_details flag
#     st.session_state.app_state = qa_testing(current_state, st.session_state.show_llm_details)
#     if st.session_state.app_state['decision'] == "passed":
#         st.success("QA Simulation Passed!")
#         st.session_state.app_state['stage'] = "Deploy" # Move to Deploy
#     else:
#         st.error(f"QA Simulation Failed: {st.session_state.app_state.get('feedback', 'Unknown error')}")
#         st.warning("Looping back - Requires Code/Test Fixes (simplified loop for demo)")
#         # In a real app, you might go back to 'Code Review' or 'Test Case Review'
#         st.session_state.app_state['stage'] = "Code Review" # Simplified loop back to Code Review
#     # Add a button to proceed after showing result
#     if st.button("Continue to Next Step"):
#         st.rerun()

elif current_state['stage'] == "QA Testing":
    st.header("10. Simulating QA Testing...")
    # Ensure you are calling qa_testing with the show_llm_details flag if you want to see the details
    st.session_state.app_state = qa_testing(current_state, st.session_state.get('show_llm_details', False))
    if st.session_state.app_state['decision'] == "passed":
        st.success("QA Simulation Passed!")
        st.session_state.app_state['stage'] = "Deploy" # Move to Deploy
        st.button("Continue to Deploy") # Optional button to trigger deploy stage
    else:
        st.error(f"QA Simulation Failed: {st.session_state.app_state.get('feedback', 'Unknown error')}")
        st.warning("QA testing failed. Looping back to 'Generate Code' to rewrite the code based on feedback.")
        # Set the stage back to "Generate Code" to trigger code regeneration
        st.session_state.app_state['stage'] = "Generate Code"
        # No need for an extra button here, the next rerun will automatically go to "Generate Code"

elif current_state['stage'] == "Deploy":
    lang = current_state.get('target_language','').capitalize()
    st.header(f"11. Simulating Deployment ({lang})...")
    # Assuming deploy function is defined elsewhere
    if 'deploy' in locals():
        st.session_state.app_state = deploy(current_state)
        st.balloons()
        # Keep stage as "Deploy" until user resets? Or move to "Deployed"
        # Let's move to Deployed to show final state message clearly
        st.session_state.app_state['stage'] = "Deployed"
        # Add button to acknowledge before showing final screen
        if st.button("Acknowledge Deployment"):
            st.rerun()
    else:
        st.warning("`deploy` function not found.")
        st.session_state.app_state['stage'] = "QA Testing"

elif current_state['stage'] == "Deployed":
    lang = current_state.get('target_language','').capitalize()
    st.header("✅ Workflow Complete!")
    st.success(f"The simulated SDLC process for {lang} finished, and the software is 'deployed'.")
    # Optionally display final artifacts again or history
    if st.button("Start New Workflow"):
        # Reset state completely
        st.session_state.app_state = SDLCState(
            stage="User Input", user_input="", target_language=None, # Reset language
            user_stories=None, design_docs=None, code=None, test_cases=None,
            decision=None, feedback=None, history=[]
        )
        st.session_state.feedback_input = ""
        st.session_state.show_feedback_box = False
        st.session_state.show_llm_details = False
        st.rerun()

# Optional: Display full state for debugging
# st.sidebar.write("Current State Details:")
# st.sidebar.json(st.session_state.app_state, expanded=False)

from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
import gradio as gr

# Set gemini pro as llm
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose=True,
                             temperature=0.5,
                             google_api_key="")

duckduckgo_search = DuckDuckGoSearchRun()

def create_crewai_setup(age, gender, disease):
    # Define Agents
    fitness_expert = Agent(
        role="Fitness Expert",
        goal=f"""Analyze the fitness requirements for a {age}-year-old {gender} with {disease} and 
                 suggest exercise routines and fitness strategies""",
        backstory=f"""Expert at understanding fitness needs, age-specific requirements, 
                      and gender-specific considerations. Skilled in developing 
                      customized exercise routines and fitness strategies.""",
        verbose=True,
        llm=llm,
        allow_delegation=True,
        tools=[duckduckgo_search],
    )

    nutritionist = Agent(
        role="Nutritionist",
        goal=f"""Assess nutritional requirements for a {age}-year-old {gender} with {disease} and
                 provide dietary recommendations""",
        backstory=f"""Knowledgeable in nutrition for different age groups and genders,
                      especially for individuals of {age} years old. Provides tailored 
                      dietary advice based on specific nutritional needs.""",
        verbose=True,
        llm=llm,
        allow_delegation=True,

    )

    doctor = Agent(
        role="Doctor",
        goal=f"""Evaluate the overall health connsiderations for a {age}-year-old {gender} with {disease} and 
                 provide recommendations for a healthy lifestyle. Pass it on to the disease_expert if you are not an expert of {disease}""",
        backstory=f"""Medical professional experienced in assessing overall health and
                      well-being. Offers recommendations for a healthy llifestyle
                      considering age, gender, and disease factors.""",
        verbose=True,
        llm=llm,
        allow_delegation=True,
    )

    # Check if the person has a disease
    if disease.lower() == "yes":
        disease_expert = Agent(
            role="Disease Expert",
            goal=f"""Provide recommendations for managing {disease}""",
            backstory=f"""Specialized in dealing with individuals having {disease}.
                          Offers tailored advice for managing the specific health condition.
                          Do not prescribe medicines but only give advice.""",

            verbose=True,
            llm=llm,
            allow_delegation=True,
        )

        disease_task = Task(
            description=f"""Provide recommendations for managing {disease}""",
            agent=disease_expert,
            llm=llm,
            expected_output="Recommendations for managing the specific disease"
        )
        health_crew = Crew(
          agents=[fitness_expert, nutritionist, doctor, disease_expert],
          tasks=[task1, task2, task3, disease_task],
          verbose=2,
          process=Process.sequential,  
        )
    else:
        # Define Tasks without Disease Expert
        task1 = Task(
            description=f"""Analyze the fitness requirements for a {age}-year-old {gender}.
                            Provide recommendations for exercise routines and fitness strategies.""",
            agent=fitness_expert,
            llm=llm,
            expected_output="Exercise routines and fitness strategies"
        )

        task2 = Task(
            description=f"""Assess nutritional requirements for a {age}-year-old {gender}.
                        Provide dietary recommendations based on specific nutritional needs.
                        Do not prescribe a medicine""",
            agent=nutritionist,
            llm=llm,
            expected_output="Dietary recommendations"

        )

        task3 = Task(
            description=f"""Evaluate overall health considerations for a {age}-year-old {gender}.
                        Provide recommendations for a healthy lifestyle.""",
                        agent=doctor,
                        llm=llm,
                        expected_output="Healthy lifestyle recommendations"
        )

        health_crew = Crew(
            agents=[fitness_expert, nutritionist, doctor],
            tasks=[task1, task2, task3],
            verbose=2,
            process=Process.sequential,
        )    

    # Create and Run the Crew
    crew_result = health_crew.kickoff()

    # Write "No disease" if the user does not have a disease
    if disease.lower() != "yes":
        crew_result += f"\n disease: {disease}"

    return crew_result

# Gradio interface
def run_crewai_app(age, gender, disease):
    crew_result = create_crewai_setup(age, gender, disease)
    return crew_result

# Custom CSS for styling
css = """
body {
    background-color: #f5f5f5;
    font-family: 'Roboto', sans-serif;
    color: #333333;
}
.gradio-container {
    background: #1b114a;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    max-width: 800px;
    margin: auto;
}
h1, h3 {
    color: #ff6f00;
    text-align: center;
}
.description {
    color: #666666;
    text-align: center;
    margin-bottom: 20px;
}
#component-0 {
    margin-bottom: 20px;
}
label {
    font-weight: bold;
    margin-bottom: 10px;
    display: block;
}
input, select, textarea {
    width: calc(100% - 20px);
    padding: 10px;
    margin-bottom: 20px;
    border: 1px solid #dddddd;
    border-radius: 4px;
    font-size: 16px;
}
button {
    background-color: #ff6f00;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    font-size: 16px;
    cursor: pointer;
}
button:hover {
    background-color: #ff8c00;
}
#output {
    margin-top: 20px;
    padding: 20px;
    border-radius: 8px;
    background-color: #1b114a;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    font-size: 16px;
    line-height: 1.5;
}
#output h2 {
    font-size: 24px;
    margin-bottom: 10px;
    color: #ff6f00;
}
#output h3 {
    font-size: 20px;
    margin-bottom: 10px;
    color: #ff8c00;
}
#output p {
    margin-bottom: 10px;
}
"""

iface = gr.Interface(
    fn=run_crewai_app,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(label="Gender", choices=["Select your gender", "Male", "Female", "Other"]),
        gr.Textbox(label="Disease", placeholder="Enter disease or 'no' if none"),
    ],
    outputs=gr.HTML(label="Output", elem_id="output"),
    title="CrewAI Health, Nutrition, and Fitness Analysis",
    description="Enter your age, gender, and whether you have any disease to receive personalized fitness, nutrition, and health strategies.",
    theme="default",
    css=css
)

iface.launch(share=True)



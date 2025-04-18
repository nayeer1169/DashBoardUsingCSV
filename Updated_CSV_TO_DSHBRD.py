import io
import json
import numpy as np
import os
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import pandas as pd
import panel as pn
import holoviews as hv
import hvplot.pandas 

os.environ['AUTOGEN_USE_DOCKER'] = '0'

with open("API_KEY.json", "r") as f:
    config_list = json.load(f)

hv.extension('bokeh')

user_proxy_agent = UserProxyAgent(
    name="User_Proxy_Agent",
    system_message=(
        "1. Collects user requirements.\n"
        "2. Validates the uploaded CSV file.\n"
        "3. Forwards relevant tasks to other agents.\n"
        "4. Ensures data consistency and correctness.\n"
        "5. Acts as the main interface between the user and the system."
    ),
    human_input_mode="NEVER",
)

code_writer_agent = AssistantAgent(
    name="Code_Writer_Agent",
    system_message=(
        "1. Generates optimized Python code for dataset processing.\n"
        "2. Ensures readability, efficiency, and correctness.\n"
        "3. Selects the best visualization type based on the data.\n"
        "4. Minimizes redundant computations and enhances performance.\n"
        "5. Follows best practices for data visualization and analysis."
    ),
    llm_config={"config_list": config_list},
)

execution_agent = AssistantAgent(
    name="Execution_Agent",
    system_message=(
        "1. Executes the generated Python code.\n"
        "2. Produces high-quality, interactive visualizations.\n"
        "3. Ensures that outputs are generated without errors.\n"
        "4. Monitors resource usage and optimizes execution speed.\n"
        "5. Integrates with Panel for dashboard rendering."
    ),
    llm_config={"config_list": config_list},
)

debugger_agent = AssistantAgent(
    name="Debugger_Agent",
    system_message=(
        "1. Identifies and corrects errors in the generated code.\n"
        "2. Ensures smooth execution without runtime issues.\n"
        "3. Provides detailed explanations for debugging steps.\n"
        "4. Implements error handling to prevent crashes.\n"
        "5. Works closely with the Execution Agent to validate fixes."
    ),
    llm_config={"config_list": config_list},
)

process_completion_agent = AssistantAgent(
    name="Process_Completion_Agent",
    system_message=(
        "1. Confirms successful data processing.\n"
        "2. Notifies the user once all tasks are completed.\n"
        "3. Ensures that the dashboard displays correctly.\n"
        "4. Provides a summary of generated visualizations.\n"
        "5. Acts as the final validation before delivering results."
    ),
    llm_config={"config_list": config_list},
)

agents = [user_proxy_agent, code_writer_agent, execution_agent, debugger_agent, process_completion_agent]
group_chat = GroupChat(agents=agents, messages=[], max_round=10)
group_chat_manager = GroupChatManager(groupchat=group_chat)

def custom_speaker_selection_func(last_speaker, groupchat):
    messages = groupchat.messages
    if len(messages) < 1:
        return user_proxy_agent
    if last_speaker is user_proxy_agent:
        return code_writer_agent
    elif last_speaker is code_writer_agent:
        return execution_agent
    elif last_speaker is execution_agent:
        if "exitcode: 1" in messages[-1]["content"]:
            return debugger_agent
        return process_completion_agent
    elif last_speaker is debugger_agent:
        return code_writer_agent
    return user_proxy_agent

def auto_select_visualizations(data):
    numerical_columns = data.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=["number"]).columns.tolist()
    visualizations = []
    
    if len(numerical_columns) >= 2:
        visualizations.append((numerical_columns[0], numerical_columns[1], "Scatter Plot"))
    
    for cat_col in categorical_columns:
        for num_col in numerical_columns:
            visualizations.append((cat_col, num_col, "Bar Chart"))
    
    for num_col in numerical_columns:
        visualizations.append((num_col, None, "Histogram"))
    
    return visualizations

def create_visualization(data, x_axis, y_axis, chart_type):
    if chart_type == "Scatter Plot":
        return data.hvplot.scatter(x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
    elif chart_type == "Bar Chart":
        return data.hvplot.bar(x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
    elif chart_type == "Histogram":
        return data[x_axis].hvplot.hist(title=f"Histogram of {x_axis}")

def create_dashboard():
    pn.extension()
    file_input = pn.widgets.FileInput(accept='.csv', width=800, height=60)
    plots_container = pn.Column(pn.layout.Spacer(height=50), align='center')
    
    user_proxy_agent.initiate_chat(group_chat_manager, message="Processing CSV and generating visualizations...")
    
    def process_file(event):
        if file_input.value:
            file_bytes = io.BytesIO(file_input.value)
            data = pd.read_csv(file_bytes)
            visualizations = auto_select_visualizations(data)
            plots_container.clear()
            hv_plots = [create_visualization(data, x_axis, y_axis, chart_type) for x_axis, y_axis, chart_type in visualizations]
            plots_container.append(pn.Column(*hv_plots, align='center'))
            process_completion_agent.initiate_chat(group_chat_manager, message="Processing complete! Your visualizations are ready.")
    
    file_input.param.watch(process_file, "value")
    
    dashboard = pn.Column(
        pn.pane.Markdown("Dashboard for CSV Visualization", styles={'font-size': '32px', 'text-align': 'center', 'color': '#222', 'font-weight': 'bold'}),
        pn.layout.Spacer(height=20),
        pn.Row(pn.pane.Markdown("Upload CSV File", styles={'font-size': '20px', 'font-weight': 'bold'}), file_input, align='center'),
        pn.layout.Spacer(height=20),
        pn.pane.Markdown("Visualizations", styles={'font-size': '20px', 'font-weight': 'bold'}),
        plots_container
    )
    
    return dashboard

dashboard = create_dashboard()
pn.serve(dashboard, port=1001, show=True)
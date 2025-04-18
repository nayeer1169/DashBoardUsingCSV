import os
import io
import json
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import pandas as pd
import panel as pn
import plotly.express as px

os.environ['AUTOGEN_USE_DOCKER'] = '0'


with open("API_KEY.json", "r") as f:
    config_list = json.load(f)


user_proxy_agent = UserProxyAgent(
    name="User_Proxy_Agent",
    system_message="You collect user requirements and forward them to the relevant agents.",
    human_input_mode="NEVER",
)

code_writer_agent = AssistantAgent(
    name="Code_Writer_Agent",
    system_message="You generate Python code for dataset processing and visualizations.",
    llm_config={"config_list": config_list},
)

execution_agent = AssistantAgent(
    name="Execution_Agent",
    system_message="You execute the generated code and produce the output visualizations.",
    llm_config={"config_list": config_list},
)

debugger_agent = AssistantAgent(
    name="Debugger_Agent",
    system_message="You debug and correct errors in the generated code.",
    llm_config={"config_list": config_list},
)

agents = [user_proxy_agent, code_writer_agent, execution_agent, debugger_agent]
group_chat = GroupChat(agents=agents, messages=[], max_round=10)
group_chat_manager = GroupChatManager(groupchat=group_chat)

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
        fig = px.scatter(data, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
    elif chart_type == "Bar Chart":
        fig = px.bar(data, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
    elif chart_type == "Histogram":
        fig = px.histogram(data, x=x_axis, title=f"Histogram of {x_axis}")
    return fig

def create_dashboard():
    pn.extension('plotly')  
    file_input = pn.widgets.FileInput(accept='.csv', width=800, height=60)
    plots_container = pn.Column()
    
    def process_file(event):
        if file_input.value:
            file_bytes = io.BytesIO(file_input.value)
            data = pd.read_csv(file_bytes)
            user_proxy_agent.initiate_chat(group_chat_manager, message="Process CSV and generate visualizations.")
            visualizations = auto_select_visualizations(data)
            plots_container.clear()
            for x_axis, y_axis, chart_type in visualizations:
                fig = create_visualization(data, x_axis, y_axis, chart_type)
                plotly_pane = pn.pane.Plotly(fig, sizing_mode="stretch_width")
                plots_container.append(plotly_pane)
    
    file_input.param.watch(process_file, "value")
    
    dashboard = pn.Column(
        pn.pane.Markdown("# 📊 Plotting Graph From CSV", styles={'font-size': '32px', 'text-align': 'center', 'color': '#222', 'font-weight': 'bold'}),
        pn.layout.Spacer(height=20),
        pn.Row(pn.pane.Markdown("## 📂 Upload CSV File", styles={'font-size': '20px', 'font-weight': 'bold', 'text-align': 'center'}), file_input, align='center'),
        pn.layout.Spacer(height=20),
        pn.pane.Markdown("## 📈 Visualizations", styles={'font-size': '20px', 'font-weight': 'bold', 'text-align': 'center'}),
        plots_container,
        css_classes=['custom-dashboard']
    )
    
    return dashboard

dashboard = create_dashboard()
pn.serve(dashboard, port=1001, show=True)

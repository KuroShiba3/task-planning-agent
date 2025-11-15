from langgraph.graph import START, StateGraph

from ...state.state import BaseState
from .nodes import plan_tasks, generate_final_answer

planner_graph = StateGraph(BaseState)
planner_graph.add_node(plan_tasks)
planner_graph.add_node(generate_final_answer)
planner_graph.add_edge(START, "plan_tasks")
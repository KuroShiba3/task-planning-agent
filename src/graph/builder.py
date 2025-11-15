from ..agents.planner.graph import planner_graph
from ..agents.websearch.graph import websearch_graph

def create_graph():
    planner_graph.add_node("websearch", websearch_graph.compile())
    planner_graph.add_edge("websearch", "generate_final_answer")

    graph = planner_graph.compile()
    return graph
from langgraph.graph import START, StateGraph

from .nodes import (
    evaluate_task_result,
    execute_search,
    generate_task_result,
    generate_search_queries,
)
from .state import WebSearchState

websearch_graph = StateGraph(WebSearchState)

websearch_graph.add_node(generate_search_queries)
websearch_graph.add_node(execute_search)
websearch_graph.add_node(generate_task_result)
websearch_graph.add_node(evaluate_task_result)

websearch_graph.add_edge(START, "generate_search_queries")
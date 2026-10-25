from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever
from app.core.rag_logging import log

settings = get_settings()

_llm = None
_web_search = None

def llm():
    global _llm
    if _llm is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        _llm = ChatOpenAI(
            model= settings.openai_model,
            temperature=0,
            api_key= settings.openai_api_key
        )
    return _llm

def web_search_tool():
    global _web_search
    if _web_search is None:
       if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
       
       _web_search= TavilySearch(
        tavily_api_key = settings.tavily_api_key,
        max_results = 5,
        topic="general",
        include_answer=True,
        include_raw_content=False)
    return _web_search

def add_trace(state: AgentState, message: str):
    return [*state.get("trace",[]), message]

# 11. Node 1 — Route the Question
#The router decides:
#- `kb`: use private knowledge base
#- `direct`: answer without retrieval
#In a real company assistant, this step may also route between many sources:
#HR docs, finance docs, product docs, SQL database, web search, etc.
def route_question(state: AgentState):
    router_llm = llm().with_structured_output(RouteDecision, method="json_mode")
  
    decision = router_llm.invoke(f"""
                You route messages for an enterprise IT support assistant.
                Use kb for questions about company IT policies, VPN, password reset, MFA, laptop setup,
                software access, security, email, devices, troubleshooting, or technology support.
                Use direct only for greetings, thanks, or casual chat that needs no company knowledge.
                Question: {state['question']}
                Return valid JSON like {{"route":"kb"}}.
                """)

    return {
        "source_used": decision.route,
        "trace": add_trace(state, f"Router -> {decision.route.upper()}")
    } 


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    if state["source_used"] == "kb":
        return "retrieve_kb"
    return "direct_answer"

def retrieve_kb(state: AgentState):
    query = state["current_query"] or state["question"]
    docs = get_retriever().invoke(query)
    return {
            "kb_docs": docs,
            "trace": add_trace(state, f"Private KB retrieval -> {len(docs)} chunks")   
            }

def grade_kb_evidence(state: AgentState):
    kb_grader_llm = llm().with_structured_output(EvidenceGrade, method="json_mode")

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in state["kb_docs"]
    )

    grade = kb_grader_llm.invoke(f"""
                                    You grade evidence for an enterprise IT support assistant.
                                    Question: {state['question']}
                                    Private company KB evidence:\n{context}
                                    Return good only if the evidence is sufficient to answer confidently and specifically.
                                    Otherwise return weak. JSON: {{"grade":"good"}} or {{"grade":"weak"}}.
                                    """)

    return {"kb_grade": grade.grade}

def decide_after_kb_grade(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    if state["kb_grade"] == "good":
        return "generate_from_kb"
    return "search_web"


#Tavily Web Search Fallback
#This node runs only when the private KB evidence is weak.
#We search the web using the original question, or the 
#current rewritten query if a rewrite already happened.
def search_web(state: AgentState):
    query = state["current_query"]

    result = web_search_tool().invoke({"query": query})

    # Tavily can return dict/list structures depending on package version.
    # Convert it into readable text for grading and generation.
    if isinstance(result, dict):
        answer = result.get("answer", "")
        results = result.get("results", [])
        lines = []
        citations=[]
        if answer:
            lines.append(f"Search answer: {answer}")

        for item in results:
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url, "type":"web"})

        web_text = "\n\n".join(lines) if lines else str(result)
    else:
        web_text = str(result)


    return {
        "web_results": web_text,
        "source_used": "web",
        "citations": citations,
        "trace": add_trace(state, "Web fallback -> Tavily search")
    }



def grade_web_evidence(state: AgentState):

    web_grader_llm = llm().with_structured_output(EvidenceGrade, method="json_mode")
    grade = web_grader_llm.invoke(f"""
                            Question: {state['question']}
                            Web evidence:\n{state['web_results']}
                            Return good if the evidence is sufficient and directly relevant; 
                            otherwise weak.
                            Return valid JSON like {{"grade":"good"}}.
                            """)
    return {"web_grade": grade.grade, "trace": add_trace(state, f"Web evidence grade → {grade.grade.upper()}")}

def decide_after_web_grade(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "answer_insufficient"

#Rewrite the Query
#If both KB and web evidence are weak, we rewrite once and then try the KB again.
#Why back to KB first?
#Because the rewritten query may help retrieve better private evidence before relying on web search again.

def rewrite_query(state: AgentState):
 
    rewritten = llm().invoke(f"""
                    Rewrite this IT support question for better private knowledge 
                    retrieval and vendor web search.
                    Preserve intent, add useful technical keywords, do not answer, 
                    return only the query.
                    Question: {state['question']}
                    """).content.strip()
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }

def generate_from_kb(state: AgentState):
    context = "\n\n".join(f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in state["kb_docs"])
    answer = llm().invoke(f"""
                    You are an enterprise IT support copilot. Answer ONLY from the private company KB below.
                    Be concise, actionable, and safe. If steps are present, present them clearly.
                    Do not invent policy details. Mention that the answer is based on the company's private knowledge base.
                    Question: {state['question']}\n\nPrivate KB:\n{context}
                    """).content
    citations = []
    seen = set()
    for d in state["kb_docs"]:
        src = d.metadata.get("source", "Private KB")
        if src not in seen:
            seen.add(src)
            citations.append({"title": src.split("/")[-1], "url": "", "type": "private_kb"})
    return {"answer": answer, "source_used": "private_kb", "citations": citations, "trace": add_trace(state, "Answer generation → PRIVATE KB")}


def generate_from_web(state: AgentState):
    answer = llm().invoke(f"""
                    You are an enterprise IT support copilot. The private company KB was insufficient.
                    Answer ONLY from the web evidence below. Clearly say this is external web information 
                    and may need IT validation before changing company-managed systems.
                    Question: {state['question']}\n\nWeb evidence:\n{state['web_results']}
                    """).content
    return {"answer": answer, "source_used": "web_search", "trace": add_trace(state, "Answer generation → WEB SEARCH")}


def direct_answer(state: AgentState):
    answer = llm().invoke(f"Respond briefly and naturally to: {state['question']}").content
    return {"answer": answer, "source_used": "direct", "trace": add_trace(state, "Direct response → no retrieval")}

def insufficient(state: AgentState):
    return {
        "answer": "I couldn't find enough reliable evidence in the company knowledge base or external search to answer confidently. Please contact the IT help desk or provide more details.",
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }


def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb_evidence,
        "search_web": search_web,
        "grade_web": grade_web_evidence,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", decide_after_kb_grade, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", decide_after_web_grade, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()

agent_graph = build_graph()

def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)
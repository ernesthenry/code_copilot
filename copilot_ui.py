import streamlit as st
from pathlib import Path
import sys
import os
import json

# Import the FastMCP server tools directly
from fastmcp_server import (
    indexer,
    index_codebase,
    search_code,
    get_file_outline,
    get_codebase_statistics,
    analyze_code,
    lint_code,
    format_code,
    run_tests,
    generate_code_documentation,
    analyze_code_complexity,
    suggest_code_improvements,
    list_files,
    read_file_content,
    get_dependencies,
)

st.set_page_config(page_title="Code Copilot UI", layout="wide")
st.title("🧑‍💻 Code Copilot - Streamlit UI")

st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to section:",
    [
        "Index Codebase",
        "Search Code",
        "File Outline",
        "Analyze Code",
        "Lint & Format",
        "Run Tests",
        "Codebase Statistics",
        "Documentation",
        "Complexity Analysis",
        "Suggest Improvements",
        "List Files",
        "Read File",
        "Dependencies",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit. Backend: FastMCP Code Copilot Server.")

if section == "Index Codebase":
    st.header("Index Codebase")
    directory = st.text_input("Directory to index", value=".")
    force = st.checkbox("Force reindex", value=False)
    if st.button("Index Now"):
        with st.spinner("Indexing codebase..."):
            result = indexer.index_directory(directory, force)
        st.success("Indexing complete!")
        st.json(result)

elif section == "Search Code":
    st.header("Search Code Symbols")
    query = st.text_input("Search query (symbol name or pattern)")
    symbol_type = st.selectbox("Symbol type", [None, "function", "class", "method", "variable"])
    file_pattern = st.text_input("File path pattern (optional)")
    limit = st.slider("Max results", 1, 100, 20)
    if st.button("Search"):
        with st.spinner("Searching..."):
            result = indexer.search_symbols(query, symbol_type, file_pattern, limit)
        if result:
            for r in result:
                with st.expander(f"{r['name']} ({r['type']}) - {r['file_path']}:{r['line_number']}"):
                    st.write(f"**Signature:** {r['signature']}")
                    st.write(f"**Docstring:** {r['docstring']}")
                    st.write(f"**Parent class:** {r['parent_class']}")
        else:
            st.warning("No results found.")

elif section == "File Outline":
    st.header("File Outline")
    file_path = st.text_input("File path")
    if st.button("Get Outline"):
        with st.spinner("Getting outline..."):
            outline = indexer.get_file_symbols(file_path)
        if outline:
            st.json(outline)
        else:
            st.warning("No symbols found or file not indexed.")

elif section == "Analyze Code":
    st.header("Analyze Python Code")
    code = st.text_area("Paste Python code here", height=200)
    file_path = st.text_input("Optional file path for context", value="<string>")
    if st.button("Analyze"):
        with st.spinner("Analyzing code..."):
            result = analyze_code(code, file_path)
        st.code(result, language="json")

elif section == "Lint & Format":
    st.header("Lint & Format Python Code")
    code = st.text_area("Paste Python code here", height=200, key="lint_code")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Lint Code"):
            with st.spinner("Linting..."):
                result = lint_code(code)
            st.text(result)
    with col2:
        if st.button("Format Code"):
            with st.spinner("Formatting..."):
                result = format_code(code)
            st.code(result, language="python")

elif section == "Run Tests":
    st.header("Run Python Tests")
    test_path = st.text_input("Test file or directory path")
    verbose = st.checkbox("Verbose output", value=False)
    if st.button("Run Tests"):
        with st.spinner("Running tests..."):
            result = run_tests(test_path, verbose)
        st.text(result)

elif section == "Codebase Statistics":
    st.header("Codebase Statistics")
    if st.button("Get Statistics"):
        with st.spinner("Fetching statistics..."):
            stats = indexer.get_codebase_stats()
        st.json(stats)

elif section == "Documentation":
    st.header("Generate Code Documentation")
    file_path = st.text_input("File path for documentation")
    if st.button("Generate Documentation"):
        with st.spinner("Generating documentation..."):
            doc = generate_code_documentation(file_path)
        st.markdown(doc)

elif section == "Complexity Analysis":
    st.header("Analyze Code Complexity")
    directory = st.text_input("Directory to analyze", value=".")
    threshold = st.slider("Complexity threshold", 1, 30, 10)
    if st.button("Analyze Complexity"):
        with st.spinner("Analyzing complexity..."):
            result = analyze_code_complexity(directory, threshold)
        st.code(result, language="json")

elif section == "Suggest Improvements":
    st.header("Suggest Code Improvements")
    file_path = st.text_input("File path for suggestions")
    improvement_type = st.selectbox("Improvement type", ["all", "performance", "style", "security", "maintainability"])
    if st.button("Suggest Improvements"):
        with st.spinner("Analyzing improvements..."):
            result = suggest_code_improvements(file_path, improvement_type)
        st.text(result)

elif section == "List Files":
    st.header("List Files in Directory")
    directory = st.text_input("Directory", value=".")
    pattern = st.text_input("Pattern (e.g. *.py)", value="*")
    if st.button("List Files"):
        with st.spinner("Listing files..."):
            result = list_files(directory, pattern)
        st.text(result)

elif section == "Read File":
    st.header("Read File Content")
    file_path = st.text_input("File path to read")
    if st.button("Read File"):
        with st.spinner("Reading file..."):
            result = read_file_content(file_path)
        st.text(result)

elif section == "Dependencies":
    st.header("Get File Dependencies")
    file_path = st.text_input("Python file path for dependency analysis")
    if st.button("Get Dependencies"):
        with st.spinner("Analyzing dependencies..."):
            result = get_dependencies(file_path)
        st.code(result, language="json")

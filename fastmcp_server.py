"""
FastMCP Code Copilot Server with Advanced Indexing
A Model Context Protocol server built with FastMCP that provides code analysis, 
file operations, development tools, and advanced code indexing capabilities.
"""

import asyncio
import json
import os
import subprocess
import ast
import tempfile
import sqlite3
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
import re

# FastMCP imports
from fastmcp import FastMCP
from fastmcp.resources import Resource
from fastmcp.tools import Tool
from fastmcp.prompts import Prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
app = FastMCP("Code Copilot Server", version="2.0.0")


@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, variable, etc.)"""
    name: str
    symbol_type: str  # 'function', 'class', 'variable', 'import', 'method'
    file_path: str
    line_number: int
    column_number: int
    end_line: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    parent_class: Optional[str] = None
    complexity: int = 0
    
@dataclass
class FileIndex:
    """Represents indexed information about a file"""
    file_path: str
    file_hash: str
    language: str
    size: int
    last_modified: float
    symbols: List[CodeSymbol]
    imports: List[str]
    dependencies: List[str]
    lines_of_code: int
    cyclomatic_complexity: int

class CodeIndexer:
    """Advanced code indexing system for codebase analysis"""
    
    def __init__(self, db_path: str = "code_index.db"):
        self.db_path = db_path
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.m': 'objective-c',
            '.sh': 'shell',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.xml': 'xml',
            '.md': 'markdown',
            '.txt': 'text'
        }
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database for code indexing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT,
                language TEXT,
                size INTEGER,
                last_modified REAL,
                lines_of_code INTEGER,
                cyclomatic_complexity INTEGER,
                indexed_at REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                symbol_type TEXT,
                file_path TEXT,
                line_number INTEGER,
                column_number INTEGER,
                end_line INTEGER,
                docstring TEXT,
                signature TEXT,
                parent_class TEXT,
                complexity INTEGER,
                FOREIGN KEY (file_path) REFERENCES files (file_path)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS imports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT,
                import_name TEXT,
                import_type TEXT,
                FOREIGN KEY (file_path) REFERENCES files (file_path)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dependencies (
                file_path TEXT,
                dependency_path TEXT,
                dependency_type TEXT,
                PRIMARY KEY (file_path, dependency_path)
            )
        ''')
        
        # Create indexes for better search performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbols_type ON symbols(symbol_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_imports_name ON imports(import_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_language ON files(language)')
        
        conn.commit()
        conn.close()
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception:
            return ""
        return hash_md5.hexdigest()
    
    def should_index_file(self, file_path: str) -> bool:
        """Check if a file should be indexed based on modification time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            file_stat = os.stat(file_path)
            current_hash = self.get_file_hash(file_path)
            
            cursor.execute(
                'SELECT file_hash, last_modified FROM files WHERE file_path = ?',
                (file_path,)
            )
            result = cursor.fetchone()
            
            if not result:
                return True
            
            stored_hash, stored_modified = result
            return current_hash != stored_hash or file_stat.st_mtime > stored_modified
            
        except Exception:
            return True
        finally:
            conn.close()
    
    def parse_python_file(self, file_path: str) -> FileIndex:
        """Parse a Python file and extract symbols"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            symbols = []
            imports = []
            
            # Extract symbols
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    signature = f"def {node.name}({', '.join(arg.arg for arg in node.args.args)})"
                    
                    symbols.append(CodeSymbol(
                        name=node.name,
                        symbol_type='function',
                        file_path=file_path,
                        line_number=node.lineno,
                        column_number=node.col_offset,
                        end_line=node.end_lineno or node.lineno,
                        docstring=docstring,
                        signature=signature,
                        complexity=self._calculate_complexity(node)
                    ))
                
                elif isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node)
                    
                    symbols.append(CodeSymbol(
                        name=node.name,
                        symbol_type='class',
                        file_path=file_path,
                        line_number=node.lineno,
                        column_number=node.col_offset,
                        end_line=node.end_lineno or node.lineno,
                        docstring=docstring,
                        signature=f"class {node.name}",
                        complexity=len(node.body)
                    ))
                    
                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_docstring = ast.get_docstring(item)
                            method_signature = f"def {item.name}({', '.join(arg.arg for arg in item.args.args)})"
                            
                            symbols.append(CodeSymbol(
                                name=item.name,
                                symbol_type='method',
                                file_path=file_path,
                                line_number=item.lineno,
                                column_number=item.col_offset,
                                end_line=item.end_lineno or item.lineno,
                                docstring=method_docstring,
                                signature=method_signature,
                                parent_class=node.name,
                                complexity=self._calculate_complexity(item)
                            ))
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            symbols.append(CodeSymbol(
                                name=target.id,
                                symbol_type='variable',
                                file_path=file_path,
                                line_number=node.lineno,
                                column_number=node.col_offset,
                                end_line=node.lineno
                            ))
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        for alias in node.names:
                            imports.append(f"{node.module}.{alias.name}")
            
            file_stat = os.stat(file_path)
            return FileIndex(
                file_path=file_path,
                file_hash=self.get_file_hash(file_path),
                language='python',
                size=file_stat.st_size,
                last_modified=file_stat.st_mtime,
                symbols=symbols,
                imports=imports,
                dependencies=[],
                lines_of_code=len(content.splitlines()),
                cyclomatic_complexity=sum(symbol.complexity for symbol in symbols)
            )
            
        except Exception as e:
            logger.error(f"Error parsing Python file {file_path}: {e}")
            return None
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function/method"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, 
                                ast.With, ast.AsyncWith, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def parse_generic_file(self, file_path: str) -> FileIndex:
        """Parse a generic file and extract basic information"""
        try:
            file_stat = os.stat(file_path)
            extension = Path(file_path).suffix.lower()
            language = self.supported_extensions.get(extension, 'unknown')
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Basic symbol extraction for non-Python files
            symbols = []
            imports = []
            
            # Extract function-like patterns
            function_patterns = {
                'javascript': r'function\s+(\w+)\s*\(',
                'typescript': r'function\s+(\w+)\s*\(',
                'java': r'(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\(',
                'cpp': r'\w+\s+(\w+)\s*\([^)]*\)\s*{',
                'c': r'\w+\s+(\w+)\s*\([^)]*\)\s*{',
            }
            
            if language in function_patterns:
                pattern = function_patterns[language]
                for match in re.finditer(pattern, content, re.MULTILINE):
                    line_num = content[:match.start()].count('\n') + 1
                    symbols.append(CodeSymbol(
                        name=match.group(-1),  # Last group is usually the function name
                        symbol_type='function',
                        file_path=file_path,
                        line_number=line_num,
                        column_number=match.start(),
                        end_line=line_num
                    ))
            
            return FileIndex(
                file_path=file_path,
                file_hash=self.get_file_hash(file_path),
                language=language,
                size=file_stat.st_size,
                last_modified=file_stat.st_mtime,
                symbols=symbols,
                imports=imports,
                dependencies=[],
                lines_of_code=len(content.splitlines()),
                cyclomatic_complexity=0
            )
            
        except Exception as e:
            logger.error(f"Error parsing generic file {file_path}: {e}")
            return None
    
    def index_file(self, file_path: str) -> bool:
        """Index a single file"""
        try:
            extension = Path(file_path).suffix.lower()
            
            if extension == '.py':
                file_index = self.parse_python_file(file_path)
            elif extension in self.supported_extensions:
                file_index = self.parse_generic_file(file_path)
            else:
                return False
            
            if not file_index:
                return False
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert or update file record
            cursor.execute('''
                INSERT OR REPLACE INTO files 
                (file_path, file_hash, language, size, last_modified, lines_of_code, cyclomatic_complexity, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_index.file_path,
                file_index.file_hash,
                file_index.language,
                file_index.size,
                file_index.last_modified,
                file_index.lines_of_code,
                file_index.cyclomatic_complexity,
                time.time()
            ))
            
            # Delete old symbols and imports
            cursor.execute('DELETE FROM symbols WHERE file_path = ?', (file_path,))
            cursor.execute('DELETE FROM imports WHERE file_path = ?', (file_path,))
            
            # Insert symbols
            for symbol in file_index.symbols:
                cursor.execute('''
                    INSERT INTO symbols 
                    (name, symbol_type, file_path, line_number, column_number, end_line, docstring, signature, parent_class, complexity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol.name,
                    symbol.symbol_type,
                    symbol.file_path,
                    symbol.line_number,
                    symbol.column_number,
                    symbol.end_line,
                    symbol.docstring,
                    symbol.signature,
                    symbol.parent_class,
                    symbol.complexity
                ))
            
            # Insert imports
            for import_name in file_index.imports:
                cursor.execute('''
                    INSERT INTO imports (file_path, import_name, import_type)
                    VALUES (?, ?, ?)
                ''', (file_path, import_name, 'import'))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Indexed {file_path} with {len(file_index.symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            return False
    
    def index_directory(self, directory: str, force_reindex: bool = False) -> Dict[str, Any]:
        """Index all files in a directory"""
        results = {
            'indexed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'total_symbols': 0,
            'processing_time': 0
        }
        
        start_time = time.time()
        
        try:
            for root, dirs, files in os.walk(directory):
                # Skip hidden directories and common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git', '.venv', 'venv']]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Skip hidden files and binary files
                    if file.startswith('.') or Path(file_path).suffix.lower() not in self.supported_extensions:
                        results['skipped_files'] += 1
                        continue
                    
                    # Check if file needs indexing
                    if not force_reindex and not self.should_index_file(file_path):
                        results['skipped_files'] += 1
                        continue
                    
                    if self.index_file(file_path):
                        results['indexed_files'] += 1
                    else:
                        results['failed_files'] += 1
        
        except Exception as e:
            logger.error(f"Error indexing directory {directory}: {e}")
        
        # Count total symbols
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM symbols')
        results['total_symbols'] = cursor.fetchone()[0]
        conn.close()
        
        results['processing_time'] = time.time() - start_time
        
        return results
    
    def search_symbols(self, query: str, symbol_type: str = None, file_pattern: str = None, limit: int = 50) -> List[Dict]:
        """Search for symbols in the codebase"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = '''
            SELECT s.name, s.symbol_type, s.file_path, s.line_number, s.signature, s.docstring, s.parent_class
            FROM symbols s
            JOIN files f ON s.file_path = f.file_path
            WHERE s.name LIKE ?
        '''
        params = [f'%{query}%']
        
        if symbol_type:
            sql += ' AND s.symbol_type = ?'
            params.append(symbol_type)
        
        if file_pattern:
            sql += ' AND s.file_path LIKE ?'
            params.append(f'%{file_pattern}%')
        
        sql += ' ORDER BY s.name LIMIT ?'
        params.append(limit)
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'name': row[0],
                'type': row[1],
                'file_path': row[2],
                'line_number': row[3],
                'signature': row[4],
                'docstring': row[5],
                'parent_class': row[6]
            }
            for row in results
        ]
    
    def get_file_symbols(self, file_path: str) -> List[Dict]:
        """Get all symbols from a specific file"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, symbol_type, line_number, signature, docstring, parent_class, complexity
            FROM symbols
            WHERE file_path = ?
            ORDER BY line_number
        ''', (file_path,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'name': row[0],
                'type': row[1],
                'line_number': row[2],
                'signature': row[3],
                'docstring': row[4],
                'parent_class': row[5],
                'complexity': row[6]
            }
            for row in results
        ]
    
    def get_codebase_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed codebase"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # File statistics
        cursor.execute('''
            SELECT language, COUNT(*) as file_count, SUM(lines_of_code) as total_lines, AVG(cyclomatic_complexity) as avg_complexity
            FROM files
            GROUP BY language
        ''')
        language_stats = cursor.fetchall()
        
        # Symbol statistics
        cursor.execute('''
            SELECT symbol_type, COUNT(*) as count
            FROM symbols
            GROUP BY symbol_type
        ''')
        symbol_stats = cursor.fetchall()
        
        # Total statistics
        cursor.execute('SELECT COUNT(*) FROM files')
        total_files = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM symbols')
        total_symbols = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(lines_of_code) FROM files')
        total_lines = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_files': total_files,
            'total_symbols': total_symbols,
            'total_lines_of_code': total_lines,
            'language_breakdown': [
                {
                    'language': row[0],
                    'files': row[1],
                    'lines': row[2],
                    'avg_complexity': round(row[3] or 0, 2)
                }
                for row in language_stats
            ],
            'symbol_breakdown': [
                {
                    'type': row[0],
                    'count': row[1]
                }
                for row in symbol_stats
            ]
        }

# Initialize the indexer
indexer = CodeIndexer()

# Add indexing tools to the FastMCP server
@app.tool()
async def index_codebase(directory: str = ".", force_reindex: bool = False) -> str:
    """
    Index the codebase for advanced searching and analysis.
    
    Args:
        directory: Directory to index (default: current directory)
        force_reindex: Force reindexing of all files
    
    Returns:
        Indexing results and statistics
    """
    try:
        directory = Path(directory).resolve()
        if not directory.exists():
            return f"Error: Directory '{directory}' does not exist"
        
        results = indexer.index_directory(str(directory), force_reindex)
        
        response = f"Codebase Indexing Results:\n"
        response += f"{'='*40}\n"
        response += f"Directory: {directory}\n"
        response += f"Indexed files: {results['indexed_files']}\n"
        response += f"Skipped files: {results['skipped_files']}\n"
        response += f"Failed files: {results['failed_files']}\n"
        response += f"Total symbols: {results['total_symbols']}\n"
        response += f"Processing time: {results['processing_time']:.2f} seconds\n"
        
        return response
        
    except Exception as e:
        return f"Error indexing codebase: {str(e)}"

@app.tool()
async def search_code(query: str, symbol_type: str = None, file_pattern: str = None, limit: int = 20) -> str:
    """
    Search for code symbols in the indexed codebase.
    
    Args:
        query: Search query (symbol name or pattern)
        symbol_type: Filter by symbol type (function, class, method, variable)
        file_pattern: Filter by file path pattern
        limit: Maximum number of results
    
    Returns:
        Search results with file locations and details
    """
    try:
        results = indexer.search_symbols(query, symbol_type, file_pattern, limit)
        
        if not results:
            return f"No symbols found matching '{query}'"
        
        response = f"Search Results for '{query}':\n"
        response += f"{'='*50}\n"
        
        for result in results:
            response += f"\n📍 {result['name']} ({result['type']})\n"
            response += f"   File: {result['file_path']}:{result['line_number']}\n"
            
            if result['parent_class']:
                response += f"   Class: {result['parent_class']}\n"
            
            if result['signature']:
                response += f"   Signature: {result['signature']}\n"
            
            if result['docstring']:
                # Truncate long docstrings
                docstring = result['docstring'][:100] + "..." if len(result['docstring']) > 100 else result['docstring']
                response += f"   Doc: {docstring}\n"
        
        return response
        
    except Exception as e:
        return f"Error searching code: {str(e)}"

@app.tool()
async def get_file_outline(file_path: str) -> str:
    """
    Get an outline of symbols in a specific file.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Structured outline of the file's symbols
    """
    try:
        symbols = indexer.get_file_symbols(file_path)
        
        if not symbols:
            return f"No symbols found in '{file_path}' (file may not be indexed)"
        
        response = f"File Outline: {file_path}\n"
        response += f"{'='*50}\n"
        
        # Group symbols by type
        symbol_groups = defaultdict(list)
        for symbol in symbols:
            symbol_groups[symbol['type']].append(symbol)
        
        for symbol_type, group in symbol_groups.items():
            response += f"\n{symbol_type.upper()}S:\n"
            for symbol in group:
                response += f"  • {symbol['name']} (line {symbol['line_number']})"
                if symbol['parent_class']:
                    response += f" in {symbol['parent_class']}"
                if symbol['complexity'] > 0:
                    response += f" [complexity: {symbol['complexity']}]"
                response += "\n"
                
                if symbol['signature']:
                    response += f"    {symbol['signature']}\n"
        
        return response
        
    except Exception as e:
        return f"Error getting file outline: {str(e)}"

@app.tool()
async def get_codebase_statistics() -> str:
    """
    Get comprehensive statistics about the indexed codebase.
    
    Returns:
        Detailed statistics about files, symbols, and complexity
    """
    try:
        stats = indexer.get_codebase_stats()
        
        response = f"Codebase Statistics:\n"
        response += f"{'='*40}\n"
        response += f"Total Files: {stats['total_files']}\n"
        response += f"Total Symbols: {stats['total_symbols']}\n"
        response += f"Total Lines of Code: {stats['total_lines_of_code']}\n\n"
        
        response += "Language Breakdown:\n"
        for lang in stats['language_breakdown']:
            response += f"  {lang['language']}: {lang['files']} files, {lang['lines']} lines"
            if lang['avg_complexity'] > 0:
                response += f", avg complexity: {lang['avg_complexity']}"
            response += "\n"
        
        response += "\nSymbol Breakdown:\n"
        for symbol in stats['symbol_breakdown']:
            response += f"  {symbol['type']}: {symbol['count']}\n"
        
        return response
        
    except Exception as e:
        return f"Error getting codebase statistics: {str(e)}"

@app.tool()
async def find_similar_functions(function_name: str, similarity_threshold: float = 0.7) -> str:
    """
    Find functions with similar names or signatures.
    
    Args:
        function_name: Name of the function to find similar matches for
        similarity_threshold: Similarity threshold (0.0 to 1.0)
    
    Returns:
        List of similar functions
    """
    try:
        # Simple similarity search based on name patterns
        conn = sqlite3.connect(indexer.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, symbol_type, file_path, line_number, signature
            FROM symbols
            WHERE symbol_type IN ('function', 'method')
            AND name != ?
        ''', (function_name,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Calculate simple similarity based on name
        similar_functions = []
        for row in results:
            name = row[0]
            # Simple similarity: count common characters
            common_chars = sum(1 for a, b in zip(function_name.lower(), name.lower()) if a == b)
            max_len = max(len(function_name), len(name))
            similarity = common_chars / max_len if max_len > 0 else 0
            
            if similarity >= similarity_threshold:
                similar_functions.append({
                    'name': name,
                    'type': row[1],
                    'file_path': row[2],
                    'line_number': row[3],
                    'signature': row[4],
                    'similarity': similarity
                })
        
        # Sort by similarity
        similar_functions.sort(key=lambda x: x['similarity'], reverse=True)
        
        if not similar_functions:
            return f"No similar functions found for '{function_name}'"
        
        response = f"Similar functions to '{function_name}':\n"
        response += f"{'='*50}\n"
        
        for func in similar_functions[:10]:  # Limit to top 10
            response += f"\n📍 {func['name']} ({func['type']}) - {func['similarity']:.2f} similarity\n"
            response += f"   File: {func['file_path']}:{func['line_number']}\n"
            if func['signature']:
                response += f"   Signature: {func['signature']}\n"
        
        return response
        
    except Exception as e:
        return f"Error finding similar functions: {str(e)}"

# Resource handlers
@app.resource("file:///{path}")
async def read_file(path: str) -> Resource:
    """Read a file resource."""
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        content = file_path.read_text(encoding='utf-8')
        return Resource(
            uri=f"file:///{file_path.absolute()}",
            name=str(file_path.name),
            description=f"File: {file_path.name}",
            text=content,
            mimeType="text/plain"
        )
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

@app.tool()
async def analyze_code(code: str, file_path: str = "<string>") -> str:
    """
    Analyze Python code for syntax errors, complexity, and structure.
    
    Args:
        code: Python code to analyze
        file_path: Optional file path for context
    
    Returns:
        JSON string with analysis results
    """
    analysis = {
        "file_path": file_path,
        "syntax_valid": False,
        "syntax_errors": [],
        "complexity_info": {},
        "structure_info": {}
    }
    
    try:
        # Parse the code to check syntax
        tree = ast.parse(code, filename=file_path)
        analysis["syntax_valid"] = True
        
        # Analyze structure
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
        
        analysis["structure_info"] = {
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "lines_of_code": len(code.splitlines())
        }
        
        # Simple complexity analysis
        analysis["complexity_info"] = {
            "num_classes": len(classes),
            "num_functions": len(functions),
            "num_imports": len(imports),
            "estimated_complexity": "Low" if len(functions) + len(classes) < 10 else "Medium"
        }
        
    except SyntaxError as e:
        analysis["syntax_errors"].append({
            "line": e.lineno,
            "message": e.msg,
            "text": e.text
        })
    
    return json.dumps(analysis, indent=2)

@app.tool()
async def run_tests(test_path: str, verbose: bool = False) -> str:
    """
    Run Python tests using pytest.
    
    Args:
        test_path: Path to test file or directory
        verbose: Enable verbose output
    
    Returns:
        Test execution results
    """
    if not Path(test_path).exists():
        return f"Error: Test path '{test_path}' does not exist"
    
    try:
        cmd = ["python", "-m", "pytest", test_path]
        if verbose:
            cmd.append("-v")
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        
        output = f"Exit code: {result.returncode}\n\n"
        output += f"STDOUT:\n{result.stdout}\n\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}"
        
        return output
    except subprocess.TimeoutExpired:
        return "Error: Test execution timed out (30s limit)"
    except Exception as e:
        return f"Error running tests: {str(e)}"

@app.tool()
async def format_code(code: str, line_length: int = 88) -> str:
    """
    Format Python code using black formatter.
    
    Args:
        code: Python code to format
        line_length: Maximum line length (default: 88)
    
    Returns:
        Formatted code or error message
    """
    try:
        # Check if black is available
        result = subprocess.run([
            "python", "-c", "import black"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            return "Error: Black formatter not installed. Install with: pip install black"
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        
        try:
            # Run black formatter
            result = subprocess.run([
                "python", "-m", "black", 
                "--line-length", str(line_length),
                tmp_path
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Read the formatted code
                formatted_code = Path(tmp_path).read_text(encoding='utf-8')
                return formatted_code
            else:
                return f"Formatting error: {result.stderr}"
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
    except subprocess.TimeoutExpired:
        return "Error: Code formatting timed out"
    except Exception as e:
        return f"Error formatting code: {str(e)}"

@app.tool()
async def git_status(path: str = ".") -> str:
    """
    Get git repository status.
    
    Args:
        path: Repository path (default: current directory)
    
    Returns:
        Git status information
    """
    try:
        # Check if it's a git repository
        result = subprocess.run([
            "git", "-C", path, "status", "--porcelain"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return f"Error: Not a git repository or git command failed\n{result.stderr}"
        
        status_output = result.stdout.strip()
        
        # Get additional git info
        branch_result = subprocess.run([
            "git", "-C", path, "branch", "--show-current"
        ], capture_output=True, text=True, timeout=5)
        
        current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
        
        output = f"Git Status for: {Path(path).absolute()}\n"
        output += f"Current branch: {current_branch}\n\n"
        
        if status_output:
            output += "Changes:\n"
            for line in status_output.split('\n'):
                if line.strip():
                    status_code = line[:2]
                    file_path = line[3:]
                    status_desc = get_git_status_description(status_code)
                    output += f"  {status_desc}: {file_path}\n"
        else:
            output += "Working tree clean - no changes detected\n"
        
        return output
        
    except subprocess.TimeoutExpired:
        return "Error: Git command timed out"
    except Exception as e:
        return f"Error getting git status: {str(e)}"

def get_git_status_description(status_code: str) -> str:
    """Convert git status codes to human-readable descriptions."""
    status_map = {
        "??": "Untracked",
        "A ": "Added",
        "M ": "Modified",
        " M": "Modified",
        "MM": "Modified (staged & unstaged)",
        "D ": "Deleted",
        " D": "Deleted",
        "R ": "Renamed",
        "C ": "Copied"
    }
    return status_map.get(status_code, f"Unknown ({status_code})")

@app.tool()
async def create_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """
    Create a new file with specified content.
    
    Args:
        file_path: Path for the new file
        content: File content
        overwrite: Whether to overwrite existing file
    
    Returns:
        Success message or error description
    """
    path = Path(file_path)
    
    try:
        if path.exists() and not overwrite:
            return f"Error: File '{file_path}' already exists. Use overwrite=true to replace it."
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        path.write_text(content, encoding='utf-8')
        
        return f"Successfully created file: {path.absolute()}\nSize: {len(content)} characters"
        
    except Exception as e:
        return f"Error creating file: {str(e)}"

@app.tool()
async def lint_code(code: str, file_path: str = "<string>") -> str:
    """
    Lint Python code using flake8.
    
    Args:
        code: Python code to lint
        file_path: Optional file path for context
    
    Returns:
        Linting results
    """
    tmp_path = None
    try:
        # Check if flake8 is available
        result = subprocess.run([
            "python", "-c", "import flake8"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            return "Error: flake8 not installed. Install with: pip install flake8"
        
        # Create a temporary file for linting
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        
        # Run flake8
        result = subprocess.run([
            "python", "-m", "flake8", tmp_path, "--max-line-length=88"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            return "No linting issues found!"
        else:
            # Replace temp file path with original file path in output
            output = result.stdout.replace(tmp_path, file_path)
            return f"Linting issues found:\n{output}"
            
    except subprocess.TimeoutExpired:
        return "Error: Linting timed out"
    except Exception as e:
        return f"Error linting code: {str(e)}"
    finally:
        # Clean up temp file
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.tool()
async def get_dependencies(file_path: str) -> str:
    """
    Extract dependencies from a Python file.
    
    Args:
        file_path: Path to the Python file
    
    Returns:
        List of dependencies
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist"
        
        content = path.read_text(encoding='utf-8')
        tree = ast.parse(content, filename=file_path)
        
        dependencies = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module.split('.')[0])
        
        # Filter out standard library modules (basic check)
        stdlib_modules = {
            'os', 'sys', 'json', 'ast', 'subprocess', 'tempfile', 'pathlib',
            'typing', 'logging', 'asyncio', 'collections', 'itertools', 're',
            'datetime', 'time', 'math', 'random', 'urllib', 'http', 'email',
            'html', 'xml', 'csv', 'configparser', 'argparse', 'unittest',
            'sqlite3', 'threading', 'multiprocessing', 'socket', 'ssl'
        }
        
        third_party_deps = dependencies - stdlib_modules
        
        result = {
            "file_path": file_path,
            "all_imports": sorted(list(dependencies)),
            "third_party_dependencies": sorted(list(third_party_deps)),
            "standard_library": sorted(list(dependencies & stdlib_modules))
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error analyzing dependencies: {str(e)}"

@app.tool()
async def read_file_content(file_path: str) -> str:
    """
    Read the content of a file.
    
    Args:
        file_path: Path to the file to read
    
    Returns:
        File content or error message
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist"
        
        content = path.read_text(encoding='utf-8')
        return f"File: {file_path}\nContent:\n{content}"
        
    except Exception as e:
        return f"Error reading file: {str(e)}"

@app.tool()
async def list_files(directory: str = ".", pattern: str = "*") -> str:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search in (default: current directory)
        pattern: File pattern to match (default: all files)
    
    Returns:
        List of matching files
    """
    try:
        path = Path(directory)
        if not path.exists():
            return f"Error: Directory '{directory}' does not exist"
        
        if not path.is_dir():
            return f"Error: '{directory}' is not a directory"
        
        files = []
        for file_path in path.glob(pattern):
            if file_path.is_file():
                files.append(str(file_path.relative_to(path)))
        
        if not files:
            return f"No files found matching pattern '{pattern}' in directory '{directory}'"
        
        return f"Files in '{directory}' matching '{pattern}':\n" + "\n".join(sorted(files))
        
    except Exception as e:
        return f"Error listing files: {str(e)}"

# Prompt handlers
@app.prompt("code-review")
async def code_review_prompt(file_path: str) -> List[Dict[str, Any]]:
    """Generate a code review prompt."""
    prompt_text = f"""Please perform a comprehensive code review of {file_path}. Focus on:

1. **Code Quality & Readability**
   - Variable and function naming
   - Code structure and organization
   - Comments and documentation

2. **Potential Issues**
   - Bugs or logical errors
   - Security vulnerabilities
   - Performance bottlenecks

3. **Best Practices**
   - Python idioms and conventions
   - Error handling
   - Testing considerations

4. **Suggestions**
   - Refactoring opportunities
   - Design pattern improvements
   - Optimization possibilities

Provide specific, actionable feedback with examples where possible."""
    
    return [{"role": "user", "content": {"type": "text", "text": prompt_text}}]

@app.prompt("debug-help")
async def debug_help_prompt(error_message: str, code_context: str = "") -> List[Dict[str, Any]]:
    """Generate a debugging help prompt."""
    prompt_text = f"""I'm encountering this error: {error_message}

Please help me debug this issue by:
1. **Explaining the Error**: What does this error mean?
2. **Root Cause Analysis**: What are the most likely causes?
3. **Debugging Steps**: Provide step-by-step debugging suggestions
4. **Solution**: Suggest fixes with code examples if applicable
5. **Prevention**: How to avoid this error in the future"""
    
    if code_context:
        prompt_text += f"\n\nHere's the relevant code context:\n```python\n{code_context}\n```"
    
    return [{"role": "user", "content": {"type": "text", "text": prompt_text}}]

@app.prompt("optimize-code")
async def optimize_code_prompt(code: str) -> List[Dict[str, Any]]:
    """Generate a code optimization prompt."""
    prompt_text = f"""Please analyze this code for performance optimization opportunities:

```python
{code}
```

Focus on:
1. **Time Complexity**: Algorithm efficiency improvements
2. **Memory Usage**: Optimization opportunities
3. **Code Structure**: Better organization and patterns
4. **Python-Specific**: Leveraging Python idioms and built-ins
5. **Scalability**: How the code performs with larger inputs

Provide specific suggestions with improved code examples and explain the performance benefits."""
    
    return [{"role": "user", "content": {"type": "text", "text": prompt_text}}]

@app.prompt("test-generation")
async def test_generation_prompt(code: str, file_path: str = "") -> List[Dict[str, Any]]:
    """Generate a test generation prompt."""
    prompt_text = f"""Please generate comprehensive unit tests for this code:

```python
{code}
```

Create tests that cover:
1. **Happy Path**: Normal operation scenarios
2. **Edge Cases**: Boundary conditions and unusual inputs
3. **Error Handling**: Exception cases and error conditions
4. **Integration**: How components work together

Use pytest framework and include:
- Clear test function names
- Appropriate fixtures if needed
- Parameterized tests for multiple scenarios
- Mocking for external dependencies
- Clear assertions with helpful error messages"""
    
    if file_path:
        prompt_text += f"\n\nFile path: {file_path}"
    
    return [{"role": "user", "content": {"type": "text", "text": prompt_text}}]

# Add these improvements to your existing code copilot server

# 1. Enhanced semantic search with vector embeddings (requires sentence-transformers)
class SemanticSearchEngine:
    """Semantic search engine for code understanding"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.embeddings = {}
        
    def generate_code_embedding(self, code_text: str, docstring: str = "") -> List[float]:
        """Generate semantic embeddings for code (mock implementation)"""
        # In a real implementation, you'd use sentence-transformers or similar
        # For now, we'll use a simple hash-based approach
        combined_text = f"{code_text} {docstring}".lower()
        words = re.findall(r'\w+', combined_text)
        # Simple word frequency vector (in production, use proper embeddings)
        return [hash(word) % 1000 for word in words[:50]]

# 2. Advanced code relationship mapping
@app.tool()
async def map_code_relationships(file_path: str) -> str:
    """Map relationships between code components (calls, inheritance, imports)"""
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=file_path)
        relationships = {
            'function_calls': [],
            'class_inheritance': [],
            'imports': [],
            'method_calls': [],
            'variable_usage': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    relationships['function_calls'].append({
                        'function': node.func.id,
                        'line': node.lineno,
                        'args_count': len(node.args)
                    })
            elif isinstance(node, ast.ClassDef):
                if node.bases:
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            relationships['class_inheritance'].append({
                                'child': node.name,
                                'parent': base.id,
                                'line': node.lineno
                            })
        
        return json.dumps(relationships, indent=2)
        
    except Exception as e:
        return f"Error mapping relationships: {str(e)}"

# 3. Code documentation generator
@app.tool()
async def generate_code_documentation(file_path: str, include_examples: bool = True) -> str:
    """Generate comprehensive documentation for a code file"""
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=file_path)
        docs = []
        
        # Extract module docstring
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            docs.append(f"# Module Documentation: {Path(file_path).name}\n\n{module_docstring}\n")
        
        # Document classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_doc = f"## Class: {node.name}\n"
                class_docstring = ast.get_docstring(node)
                if class_docstring:
                    class_doc += f"{class_docstring}\n"
                
                # Document methods
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if methods:
                    class_doc += "\n### Methods:\n"
                    for method in methods:
                        method_sig = f"def {method.name}({', '.join(arg.arg for arg in method.args.args)})"
                        class_doc += f"- `{method_sig}`\n"
                        method_docstring = ast.get_docstring(method)
                        if method_docstring:
                            class_doc += f"  {method_docstring.split('.')[0]}.\n"
                
                docs.append(class_doc)
                
            elif isinstance(node, ast.FunctionDef) and not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) if hasattr(parent, 'body') and node in getattr(parent, 'body', [])):
                func_doc = f"## Function: {node.name}\n"
                func_sig = f"def {node.name}({', '.join(arg.arg for arg in node.args.args)})"
                func_doc += f"```python\n{func_sig}\n```\n"
                
                func_docstring = ast.get_docstring(node)
                if func_docstring:
                    func_doc += f"{func_docstring}\n"
                
                docs.append(func_doc)
        
        return "\n".join(docs)
        
    except Exception as e:
        return f"Error generating documentation: {str(e)}"

# 4. Code complexity analyzer
@app.tool()
async def analyze_code_complexity(directory: str = ".", threshold: int = 10) -> str:
    """Analyze code complexity across the codebase and identify refactoring candidates"""
    try:
        complexity_report = {
            'high_complexity_functions': [],
            'refactoring_candidates': [],
            'overall_stats': {}
        }
        
        total_functions = 0
        total_complexity = 0
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        tree = ast.parse(content, filename=file_path)
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                complexity = calculate_detailed_complexity(node)
                                total_functions += 1
                                total_complexity += complexity
                                
                                if complexity > threshold:
                                    complexity_report['high_complexity_functions'].append({
                                        'function': node.name,
                                        'file': file_path,
                                        'line': node.lineno,
                                        'complexity': complexity,
                                        'recommendation': get_complexity_recommendation(complexity)
                                    })
                    
                    except Exception as e:
                        logger.warning(f"Error analyzing {file_path}: {e}")
        
        complexity_report['overall_stats'] = {
            'total_functions': total_functions,
            'average_complexity': round(total_complexity / total_functions, 2) if total_functions > 0 else 0,
            'functions_above_threshold': len(complexity_report['high_complexity_functions'])
        }
        
        return json.dumps(complexity_report, indent=2)
        
    except Exception as e:
        return f"Error analyzing complexity: {str(e)}"

def calculate_detailed_complexity(node: ast.AST) -> int:
    """Calculate detailed cyclomatic complexity"""
    complexity = 1
    nested_level = 0
    
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            complexity += 1 + nested_level
        elif isinstance(child, ast.Try):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, ast.With, ast.AsyncWith):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
        elif isinstance(child, ast.Lambda):
            complexity += 1
    
    return complexity

def get_complexity_recommendation(complexity: int) -> str:
    """Get refactoring recommendations based on complexity"""
    if complexity > 20:
        return "Critical: Consider breaking into smaller functions"
    elif complexity > 15:
        return "High: Extract methods or use strategy pattern"
    elif complexity > 10:
        return "Medium: Consider refactoring for better readability"
    else:
        return "Low: Acceptable complexity"

# 5. Team onboarding assistant
@app.tool()
async def generate_onboarding_guide(focus_area: str = "overview") -> str:
    """Generate personalized onboarding guide for new developers"""
    try:
        stats = indexer.get_codebase_stats()
        
        guide = f"""# Developer Onboarding Guide - {focus_area.title()}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Codebase Overview
- **Total Files**: {stats['total_files']}
- **Total Lines**: {stats['total_lines_of_code']}
- **Languages**: {', '.join([lang['language'] for lang in stats['language_breakdown']])}

## Getting Started Checklist
□ Clone the repository
□ Set up development environment
□ Run `index_codebase` to build code index
□ Explore main entry points
□ Review architecture documentation
□ Run existing tests

## Key Areas to Explore
"""
        
        if focus_area == "overview":
            guide += """
### 1. Main Entry Points
Use `search_code` to find:
- `main()` functions
- `__init__.py` files
- Configuration files

### 2. Core Components
Look for:
- Base classes and interfaces
- Utility modules
- Configuration management

### 3. Testing Strategy
- Test files location
- Testing frameworks used
- Coverage reports
"""
        
        elif focus_area == "architecture":
            guide += """
### 1. Design Patterns
- Review class hierarchies
- Identify design patterns used
- Understand data flow

### 2. Module Dependencies
- Use `map_code_relationships` for dependency analysis
- Identify circular dependencies
- Review import patterns

### 3. Configuration Management
- Environment variables
- Configuration files
- Feature flags
"""
        
        elif focus_area == "development":
            guide += """
### 1. Development Workflow
- Branch naming conventions
- Code review process
- Deployment pipeline

### 2. Coding Standards
- Style guide compliance
- Documentation requirements
- Error handling patterns

### 3. Debugging Tips
- Common issues and solutions
- Logging strategies
- Performance monitoring
"""
        
        guide += """
## Useful Commands
- `search_code "function_name"` - Find specific functions
- `get_file_outline "path/to/file.py"` - See file structure
- `analyze_code_complexity` - Find complex code
- `generate_code_documentation "file.py"` - Create docs
- `map_code_relationships "file.py"` - Understand dependencies

## Next Steps
1. Start with high-level architecture overview
2. Dive into specific modules of interest
3. Review test cases for understanding
4. Make small contributions to build confidence
"""
        
        return guide
        
    except Exception as e:
        return f"Error generating onboarding guide: {str(e)}"

# 6. Code improvement suggestions
@app.tool()
async def suggest_code_improvements(file_path: str, improvement_type: str = "all") -> str:
    """Suggest specific improvements for a code file"""
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=file_path)
        suggestions = []
        
        # Performance improvements
        if improvement_type in ["all", "performance"]:
            suggestions.extend(analyze_performance_issues(tree, content))
        
        # Code style improvements
        if improvement_type in ["all", "style"]:
            suggestions.extend(analyze_style_issues(tree, content))
        
        # Security improvements
        if improvement_type in ["all", "security"]:
            suggestions.extend(analyze_security_issues(tree, content))
        
        # Maintainability improvements
        if improvement_type in ["all", "maintainability"]:
            suggestions.extend(analyze_maintainability_issues(tree, content))
        
        if not suggestions:
            return "No specific improvements found. Code looks good!"
        
        result = f"Code Improvement Suggestions for {file_path}:\n\n"
        for i, suggestion in enumerate(suggestions, 1):
            result += f"{i}. **{suggestion['type']}** (Line {suggestion['line']})\n"
            result += f"   Issue: {suggestion['issue']}\n"
            result += f"   Suggestion: {suggestion['suggestion']}\n"
            if suggestion.get('example'):
                result += f"   Example: {suggestion['example']}\n"
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing code improvements: {str(e)}"

def analyze_performance_issues(tree: ast.AST, content: str) -> List[Dict]:
    """Analyze performance-related issues"""
    issues = []
    
    for node in ast.walk(tree):
        # Check for string concatenation in loops
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                    if isinstance(child.target, ast.Name):
                        issues.append({
                            'type': 'Performance',
                            'line': node.lineno,
                            'issue': 'String concatenation in loop',
                            'suggestion': 'Use list.append() and join() instead',
                            'example': 'result = []; result.append(item); final = "".join(result)'
                        })
        
        # Check for inefficient list operations
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'append':
                parent = node.parent if hasattr(node, 'parent') else None
                if isinstance(parent, (ast.For, ast.While)):
                    issues.append({
                        'type': 'Performance',
                        'line': node.lineno,
                        'issue': 'Consider list comprehension',
                        'suggestion': 'Use list comprehension for better performance',
                        'example': 'result = [process(item) for item in items]'
                    })
    
    return issues

def analyze_style_issues(tree: ast.AST, content: str) -> List[Dict]:
    """Analyze code style issues"""
    issues = []
    
    for node in ast.walk(tree):
        # Check for long functions
        if isinstance(node, ast.FunctionDef):
            if node.end_lineno and (node.end_lineno - node.lineno) > 50:
                issues.append({
                    'type': 'Style',
                    'line': node.lineno,
                    'issue': f'Function {node.name} is too long ({node.end_lineno - node.lineno} lines)',
                    'suggestion': 'Consider breaking into smaller functions',
                    'example': 'Extract logical blocks into separate helper functions'
                })
        
        # Check for missing docstrings
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if not ast.get_docstring(node):
                issues.append({
                    'type': 'Documentation',
                    'line': node.lineno,
                    'issue': f'{node.__class__.__name__[:-3]} {node.name} missing docstring',
                    'suggestion': 'Add descriptive docstring',
                    'example': '"""Brief description of purpose and parameters."""'
                })
    
    return issues

def analyze_security_issues(tree: ast.AST, content: str) -> List[Dict]:
    """Analyze potential security issues"""
    issues = []
    
    for node in ast.walk(tree):
        # Check for eval() usage
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'eval':
            issues.append({
                'type': 'Security',
                'line': node.lineno,
                'issue': 'Use of eval() function',
                'suggestion': 'Use ast.literal_eval() or safer alternatives',
                'example': 'ast.literal_eval(string) for safe evaluation'
            })
        
        # Check for shell command injection
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ['system', 'popen', 'call']:
                issues.append({
                    'type': 'Security',
                    'line': node.lineno,
                    'issue': 'Potential shell injection vulnerability',
                    'suggestion': 'Use subprocess with shell=False and validate inputs',
                    'example': 'subprocess.run([cmd, arg1, arg2], shell=False)'
                })
    
    return issues

def analyze_maintainability_issues(tree: ast.AST, content: str) -> List[Dict]:
    """Analyze maintainability issues"""
    issues = []
    
    for node in ast.walk(tree):
        # Check for too many parameters
        if isinstance(node, ast.FunctionDef) and len(node.args.args) > 5:
            issues.append({
                'type': 'Maintainability',
                'line': node.lineno,
                'issue': f'Function {node.name} has too many parameters ({len(node.args.args)})',
                'suggestion': 'Consider using a configuration object or dataclass',
                'example': 'def func(config: Config): ... where Config is a dataclass'
            })
        
        # Check for deeply nested code
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            depth = get_nesting_depth(node)
            if depth > 3:
                issues.append({
                    'type': 'Maintainability',
                    'line': node.lineno,
                    'issue': f'Code is too deeply nested (depth: {depth})',
                    'suggestion': 'Extract nested logic into separate functions',
                    'example': 'Use early returns or extract methods to reduce nesting'
                })
    
    return issues

def get_nesting_depth(node: ast.AST, current_depth: int = 0) -> int:
    """Calculate nesting depth of a node"""
    max_depth = current_depth
    
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            child_depth = get_nesting_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
    
    return max_depth

# 7. Interactive code explorer
@app.tool()
async def explore_code_interactively(query: str, context: str = "") -> str:
    """Interactive code exploration with context-aware responses"""
    try:
        # Parse the query to understand what the user wants to explore
        query_lower = query.lower()
        
        if "how does" in query_lower or "how to" in query_lower:
            # User wants to understand functionality
            return await explain_functionality(query, context)
        
        elif "find" in query_lower or "search" in query_lower:
            # User wants to find something
            return await intelligent_search(query, context)
        
        elif "improve" in query_lower or "optimize" in query_lower:
            # User wants improvement suggestions
            return await suggest_improvements_contextual(query, context)
        
        elif "example" in query_lower or "show me" in query_lower:
            # User wants examples
            return await provide_code_examples(query, context)
        
        else:
            # General exploration
            return await general_code_exploration(query, context)
    
    except Exception as e:
        return f"Error in interactive exploration: {str(e)}"

async def explain_functionality(query: str, context: str) -> str:
    """Explain how specific functionality works"""
    # Extract function/class names from query
    words = re.findall(r'\b\w+\b', query)
    code_elements = []
    
    for word in words:
        if word not in ['how', 'does', 'work', 'function', 'class', 'method']:
            results = indexer.search_symbols(word, limit=5)
            code_elements.extend(results)
    
    if not code_elements:
        return "I couldn't find specific code elements to explain. Try being more specific with function or class names."
    
    explanation = "Here's how the functionality works:\n\n"
    
    for element in code_elements[:3]:  # Limit to top 3 results
        explanation += f"## {element['name']} ({element['type']})\n"
        explanation += f"**Location**: {element['file_path']}:{element['line_number']}\n"
        
        if element['signature']:
            explanation += f"**Signature**: `{element['signature']}`\n"
        
        if element['docstring']:
            explanation += f"**Purpose**: {element['docstring'][:200]}...\n"
        
        # Get related functions
        related = indexer.search_symbols(element['name'][:-2] if element['name'].endswith('ed') else element['name'], limit=3)
        if related:
            explanation += f"**Related functions**: {', '.join([r['name'] for r in related[:3]])}\n"
        
        explanation += "\n"
    
    return explanation

async def intelligent_search(query: str, context: str) -> str:
    """Perform intelligent search based on user intent"""
    search_terms = re.findall(r'\b\w+\b', query)
    search_terms = [term for term in search_terms if term not in ['find', 'search', 'for', 'the', 'in', 'with']]
    
    if not search_terms:
        return "Please provide specific terms to search for."
    
    results = []
    for term in search_terms:
        results.extend(indexer.search_symbols(term, limit=10))
    
    # Remove duplicates and sort by relevance
    unique_results = {}
    for result in results:
        key = f"{result['file_path']}:{result['line_number']}"
        if key not in unique_results:
            unique_results[key] = result
    
    if not unique_results:
        return f"No results found for: {', '.join(search_terms)}"
    
    response = f"Search results for: {', '.join(search_terms)}\n\n"
    
    for i, result in enumerate(list(unique_results.values())[:10], 1):
        response += f"{i}. **{result['name']}** ({result['type']})\n"
        response += f"   📁 {result['file_path']}:{result['line_number']}\n"
        
        if result['signature']:
            response += f"   🔧 {result['signature']}\n"
        
        if result['docstring']:
            doc_preview = result['docstring'][:100] + "..." if len(result['docstring']) > 100 else result['docstring']
            response += f"   📝 {doc_preview}\n"
        
        response += "\n"
    
    return response

async def suggest_improvements_contextual(query: str, context: str) -> str:
    """Provide contextual improvement suggestions"""
    # Extract file paths or function names from query
    file_matches = re.findall(r'[\w/]+\.py', query)
    
    if file_matches:
        file_path = file_matches[0]
        return await suggest_code_improvements(file_path, "all")
    
    # If no file specified, provide general improvement suggestions
    return """Here are some general improvement strategies:

1. **Code Complexity**: Use `analyze_code_complexity` to find complex functions
2. **Documentation**: Use `generate_code_documentation` to create docs
3. **Performance**: Look for loops with string concatenation or inefficient operations
4. **Security**: Check for eval(), shell commands, or SQL injection risks
5. **Maintainability**: Reduce nesting, limit function parameters, add type hints

Use specific file paths for targeted suggestions."""

async def provide_code_examples(query: str, context: str) -> str:
    """Provide relevant code examples"""
    # Search for examples based on query
    search_terms = re.findall(r'\b\w+\b', query)
    examples = []
    
    for term in search_terms:
        if term not in ['example', 'show', 'me', 'how', 'to']:
            results = indexer.search_symbols(term, limit=3)
            examples.extend(results)
    
    if not examples:
        return "I couldn't find specific examples. Try searching for function or class names."
    
    response = "Here are some relevant code examples:\n\n"
    
    for i, example in enumerate(examples[:3], 1):
        response += f"## Example {i}: {example['name']}\n"
        response += f"**File**: {example['file_path']}:{example['line_number']}\n"
        
        if example['signature']:
            response += f"```python\n{example['signature']}\n```\n"
        
        if example['docstring']:
            response += f"**Description**: {example['docstring'][:150]}...\n"
        
        response += "\n"
    
    return response

async def general_code_exploration(query: str, context: str) -> str:
    """General code exploration and analysis"""
    stats = indexer.get_codebase_stats()
    
    response = f"Code Exploration Results:\n\n"
    response += f"**Codebase Overview**:\n"
    response += f"- Total files: {stats['total_files']}\n"
    response += f"- Total symbols: {stats['total_symbols']}\n"
    response += f"- Languages: {', '.join([lang['language'] for lang in stats['language_breakdown']])}\n\n"
    
    response += f"**Query**: {query}\n"
    response += f"**Context**: {context}\n\n"
    
    # Provide relevant suggestions based on query
    if any(word in query.lower() for word in ['test', 'testing']):
        response += "**Testing**: Look for files with 'test_' prefix or in 'tests/' directory\n"
    
    if any(word in query.lower() for word in ['config', 'settings']):
        response += "**Configuration**: Search for 'config', 'settings', or '.env' files\n"
    
    if any(word in query.lower() for word in ['main', 'entry', 'start']):
        response += "**Entry Points**: Look for 'main()', '__main__', or 'app.py' files\n"
    
    response += "\n**Suggested Commands**:\n"
    response += "- `search_code 'function_name'` - Find specific functions\n"
    response += "- `get_file_outline 'file.py'` - See file structure\n"
    response += "- `generate_onboarding_guide` - Get started guide\n"
    
    return response

# 8. Add the remaining tools and prompts from the original file
# (Complete the explore_codebase_prompt that was cut off)

@app.prompt("explore-codebase")
async def explore_codebase_prompt(focus_area: str = "overview") -> List[Dict[str, Any]]:
    """Generate a codebase exploration prompt."""
    prompt_text = f"""I'd like to explore this codebase with focus on: {focus_area}

Please help me understand:
1. **Architecture Overview**: High-level structure and organization
2. **Key Components**: Main modules, classes, and functions
3. **Dependencies**: External libraries and internal relationships
4. **Data Flow**: How data moves through the system
5. **Entry Points**: Main execution paths and interfaces
6. **Design Patterns**: Architectural patterns used
7. **Code Quality**: Overall maintainability and structure

Focus Areas Available:
- overview: General codebase structure
- architecture: System design and patterns
- dependencies: Library usage and module relationships
- complexity: Code complexity analysis
- testing: Test coverage and strategies
- performance: Performance-related aspects
- security: Security considerations

Please start by indexing the codebase if not already done, then provide insights based on the focus area."""
    
    return [{"role": "user", "content": {"type": "text", "text": prompt_text}}]

@app.tool()
async def parse_form_docstring(file_path: str, class_name: str) -> str:
    """
    Parse the docstring of a form class to extract structured payload and model info.
    Args:
        file_path: Path to the Python file containing the class
        class_name: Name of the class to parse
    Returns:
        JSON string with 'expected_payload' and 'associated_model' fields.
    """
    import ast
    import re
    from pathlib import Path
    import json

    def extract_section(docstring, section):
        # Find section header and extract lines until next section or end
        pattern = rf"^\s*-?\s*{re.escape(section)}:?\s*$"
        lines = docstring.splitlines()
        start = None
        for i, line in enumerate(lines):
            if section.lower() in line.lower():
                start = i
                break
        if start is None:
            return []
        # Collect lines until next section or empty line
        result = []
        for line in lines[start+1:]:
            if line.strip().startswith('-') or line.strip() == '' or re.match(r"^[A-Z][a-z]+", line.strip()):
                if result and not line.strip().startswith('-'):
                    break
            if line.strip().startswith('-'):
                result.append(line.strip())
        return result

    def parse_expected_payload(lines):
        # Example: - language_model (str): The language model to be used (required).
        payload = []
        for line in lines:
            m = re.match(r"-\s*(\w+)\s*\(([^)]+)\):\s*(.*)", line)
            if m:
                name, typ, desc = m.groups()
                payload.append({"name": name, "type": typ, "description": desc})
        return payload

    def parse_associated_model(lines):
        # Example: - ConversationalAISettings: ...
        for line in lines:
            m = re.match(r"-\s*([\w.]+):?\s*(.*)", line)
            if m:
                return m.group(1)
        return None

    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"error": f"File '{file_path}' does not exist"})
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                docstring = ast.get_docstring(node)
                if not docstring:
                    return json.dumps({"error": f"No docstring found for class '{class_name}'"})
                payload_lines = extract_section(docstring, "Expected Payload")
                model_lines = extract_section(docstring, "Associated Model")
                expected_payload = parse_expected_payload(payload_lines)
                associated_model = parse_associated_model(model_lines)
                return json.dumps({
                    "expected_payload": expected_payload,
                    "associated_model": associated_model
                }, indent=2)
        return json.dumps({"error": f"Class '{class_name}' not found in file"})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Main execution
if __name__ == "__main__":
    app.run()
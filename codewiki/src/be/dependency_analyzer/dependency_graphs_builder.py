from typing import Dict, List, Any
import os
from codewiki.src.config import Config
from codewiki.src.be.dependency_analyzer.ast_parser import DependencyParser
from codewiki.src.be.dependency_analyzer.topo_sort import build_graph_from_components, get_leaf_nodes
from codewiki.src.utils import file_manager

import logging
logger = logging.getLogger(__name__)


class DependencyGraphBuilder:
    """Handles dependency analysis and graph building."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def build_dependency_graph(self) -> tuple[Dict[str, Any], List[str]]:
        """
        Build and save dependency graph, returning components and leaf nodes.
        
        Returns:
            Tuple of (components, leaf_nodes)
        """
        # Ensure output directory exists
        file_manager.ensure_directory(self.config.dependency_graph_dir)

        # Prepare dependency graph path
        repo_name = os.path.basename(os.path.normpath(self.config.repo_path))
        sanitized_repo_name = ''.join(c if c.isalnum() else '_' for c in repo_name)
        dependency_graph_path = os.path.join(
            self.config.dependency_graph_dir, 
            f"{sanitized_repo_name}_dependency_graph.json"
        )
        filtered_folders_path = os.path.join(
            self.config.dependency_graph_dir, 
            f"{sanitized_repo_name}_filtered_folders.json"
        )

        parser = DependencyParser(self.config.repo_path)

        filtered_folders = None
        # if os.path.exists(filtered_folders_path):
        #     logger.debug(f"Loading filtered folders from {filtered_folders_path}")
        #     filtered_folders = file_manager.load_json(filtered_folders_path)
        # else:
        #     # Parse repository
        #     filtered_folders = parser.filter_folders()
        #     # Save filtered folders
        #     file_manager.save_json(filtered_folders, filtered_folders_path)

        # Parse repository
        components = parser.parse_repository(filtered_folders)
        
        # Save dependency graph
        parser.save_dependency_graph(dependency_graph_path)
        
        # Build graph for traversal
        graph = build_graph_from_components(components)
        
        # Get leaf nodes (already filtered by get_leaf_nodes to only include valid class/interface/struct)
        leaf_nodes = get_leaf_nodes(graph, components)

        return components, leaf_nodes
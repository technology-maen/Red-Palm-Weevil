"""
Red Palm Weevil Spread Simulation.

Usage: streamlit run graph_app.py

This app simulates and visualizes the spread of Red Palm Weevil (RPW) across a palm farm.
It identifies the most likely infection paths based on infection percentages between palms.
"""

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from typing import List, Dict

# Page configuration
st.set_page_config(page_title="🦗 Red Palm Weevil Spread Simulator", layout="wide")

# Title and description
st.title("🦗 Red Palm Weevil Spread Simulator")
st.markdown("""
This application simulates and visualizes the spread of **Red Palm Weevil (RPW)** across your palm farm.
It identifies the most likely infection paths where the path with the **highest cumulative infection risk** represents
the most probable spread corridor. Each connection between palms has an **infection percentage** (0-100%)
that represents the likelihood of RPW transmission.
""")

# --- Initialize Session State ---
if 'graph' not in st.session_state:
    st.session_state.graph = None
    st.session_state.edges_data = None
    st.session_state.edit_weights = {}
    st.session_state.vertices = ['Palm A', 'Palm B', 'Palm C', 'Palm D', 'Palm E']
    st.session_state.vertex_positions = {
        'Palm A': (0.5, 0.8),
        'Palm B': (0.25, 0.5),
        'Palm C': (0.75, 0.5),
        'Palm D': (0.33, 0.15),
        'Palm E': (0.67, 0.15)
    }

# --- Graph Setup ---
def create_graph_from_edges(edges_list):
    """Create the graph from a list of edges"""
    G = nx.Graph()
    
    # Add vertices from session state
    G.add_nodes_from(st.session_state.vertices)
    
    # Add edges to graph
    for u, v, weight, label in edges_list:
        G.add_edge(u, v, weight=weight, label=label)
    
    return G

def get_default_edges():
    """Return default edges representing infection transmission risk (0-100%)"""
    return [
        ('Palm A', 'Palm B', 35, 'Transmission 1'),
        ('Palm A', 'Palm C', 65, 'Transmission 2'),
        ('Palm B', 'Palm D', 25, 'Transmission 3'),
        ('Palm C', 'Palm D', 55, 'Transmission 4'),
        ('Palm C', 'Palm E', 75, 'Transmission 5'),
        ('Palm D', 'Palm E', 40, 'Transmission 6')
    ]

# Initialize graph data
if st.session_state.edges_data is None:
    st.session_state.edges_data = get_default_edges()

st.session_state.graph = create_graph_from_edges(st.session_state.edges_data)

def get_vertex_positions():
    """Get positions for vertices from session state"""
    return st.session_state.vertex_positions

def find_all_paths_from_vertex(graph, start_vertex) -> List[Dict]:
    """Find all simple paths from start vertex to all other vertices"""
    all_paths_with_weights = []
    
    # Get all other vertices (destinations)
    other_vertices = [v for v in graph.nodes() if v != start_vertex]
    
    for end_vertex in other_vertices:
        # Find all simple paths from start to this end vertex
        all_paths = list(nx.all_simple_paths(graph, start_vertex, end_vertex))
        
        for path in all_paths:
            total_weight = 0
            edge_sequence = []
            
            # Calculate total weight for this path
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                weight = graph[u][v]['weight']
                label = graph[u][v]['label']
                total_weight += weight
                edge_sequence.append(f"{label}({u}→{v}, w={weight})")
            
            all_paths_with_weights.append({
                'start': start_vertex,
                'end': end_vertex,
                'path': path,
                'total_weight': total_weight,
                'edge_sequence': edge_sequence
            })
    
    # Sort by total weight (descending - longest first)
    all_paths_with_weights.sort(key=lambda x: x['total_weight'], reverse=True)
    return all_paths_with_weights

def draw_graph(graph, selected_vertex=None):
    """Draw the RPW spread network visualization"""
    if graph.number_of_edges() == 0:
        st.warning("🦟 No transmission paths in the network. Add connections first!")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    pos = get_vertex_positions()
    
    # Draw nodes with red/orange theme for infection
    nx.draw_networkx_nodes(graph, pos, node_color='#ff6b6b', node_size=1500,
                           edgecolors='#c92a2a', linewidths=3, ax=ax)
    
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold',
                            font_color='white', ax=ax)
    
    # Draw edges with color intensity based on infection percentage
    nx.draw_networkx_edges(graph, pos, width=2.5, alpha=0.7, 
                          edge_color='#d63031', ax=ax)
    
    # Create edge labels with transmission risk percentages
    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        infection_pct = data['weight']
        edge_labels[(u, v)] = f"{data['label']}\nRisk: {infection_pct}%"
    
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=9,
                                 font_weight='bold', 
                                 bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='#ffe0e0', alpha=0.9), ax=ax)
    
    # Add network information as text
    palm_names = ", ".join(graph.nodes())
    info_text = f"🦟 RPW Farm Spread Network\nFarm Palms: {palm_names}"
    ax.text(0.5, -0.05, info_text, ha='center', fontsize=11, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#ffe0e0', alpha=0.7))
    
    ax.set_title('🦟 Red Palm Weevil Spread Risk Network', fontsize=16, fontweight='bold', color='#c92a2a')
    ax.axis('off')
    plt.tight_layout()
    
    return fig

# --- Main Content ---

# Create the graph
G = st.session_state.graph
vertices = st.session_state.vertices

# Sidebar for user input
st.sidebar.header("🦟 Simulation")
selected_vertex = st.sidebar.selectbox(
    "Select primary infection source:",
    vertices,
    help="Choose the palm from which to trace RPW spread paths"
)

# --- Vertex Management Section ---
st.sidebar.markdown("---")
st.sidebar.header("🌴 Farm Management")

tab_add_v, tab_remove_v = st.sidebar.tabs(["Add Palm", "Remove Palm"])

with tab_add_v:
    st.write("**🌴 Add a new palm to the farm:**")
    new_vertex = st.text_input("Palm identifier (e.g., Palm F):", key="new_vertex_input")
    col1, col2 = st.columns(2)
    
    with col1:
        x_pos = st.number_input("Position (X, left to right):", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="x_pos")
    
    with col2:
        y_pos = st.number_input("Position (Y, top to bottom):", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="y_pos")
    
    if st.button("🌱 Add Palm to Farm"):
        if not new_vertex.strip():
            st.error("Please enter a palm identifier!")
        elif new_vertex in st.session_state.vertices:
            st.error(f"🦟 '{new_vertex}' already exists in the farm!")
        else:
            st.session_state.vertices.append(new_vertex)
            st.session_state.vertex_positions[new_vertex] = (x_pos, y_pos)
            st.session_state.graph = create_graph_from_edges(st.session_state.edges_data)
            st.success(f"✅ '{new_vertex}' added to farm!")
            st.rerun()

with tab_remove_v:
    st.write("**🗑️ Remove a palm from the farm:**")
    
    if len(st.session_state.vertices) == 0:
        st.info("No palms to remove!")
    else:
        vertex_to_remove = st.selectbox("Select palm to remove:", st.session_state.vertices, key="remove_vertex")
        
        if st.button("🗑️ Remove Palm from Farm"):
            # Remove the vertex
            st.session_state.vertices.remove(vertex_to_remove)
            
            # Remove its position
            if vertex_to_remove in st.session_state.vertex_positions:
                del st.session_state.vertex_positions[vertex_to_remove]
            
            # Remove all edges connected to this vertex
            new_edges = []
            for u, v, w, label in st.session_state.edges_data:
                if u != vertex_to_remove and v != vertex_to_remove:
                    new_edges.append((u, v, w, label))
            
            st.session_state.edges_data = new_edges
            st.session_state.graph = create_graph_from_edges(st.session_state.edges_data)
            st.success(f"✅ '{vertex_to_remove}' removed from farm!")
            st.rerun()

# --- Edge Management Section ---
st.sidebar.markdown("---")
st.sidebar.header("🦟 Transmission Paths")

tab_edit, tab_add, tab_remove = st.sidebar.tabs(["Edit Risk %", "Add Transmission", "Remove Transmission"])

with tab_edit:
    st.write("**🦟 Edit RPW transmission risk percentage:**")
    
    # Create a form for editing weights
    with st.form(key="edit_weights_form"):
        edges_list = st.session_state.edges_data
        updated_values = {}
        
        for idx, (u, v, weight, label) in enumerate(edges_list):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"{label}: {u} ↔ {v}")
            
            with col2:
                updated_values[idx] = st.number_input(
                    "Risk %",
                    value=weight,
                    min_value=0,
                    max_value=100,
                    key=f"weight_input_{idx}",
                    label_visibility="collapsed"
                )
        
        submit_edit = st.form_submit_button("💾 Update Risk %")
    
    if submit_edit:
        new_edges = []
        for idx, (u, v, weight, label) in enumerate(edges_list):
            new_edges.append((u, v, updated_values[idx], label))
        
        st.session_state.edges_data = new_edges
        st.session_state.graph = create_graph_from_edges(st.session_state.edges_data)
        st.success("✅ Transmission risk updated!")
        st.rerun()

with tab_add:
    st.write("**� Add RPW transmission path between palms:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_u = st.selectbox("From:", st.session_state.vertices, key="add_from")
    
    with col2:
        new_v = st.selectbox("To:", st.session_state.vertices, key="add_to")
    
    with col3:
        new_weight = st.number_input("Risk %:", min_value=0, max_value=100, value=50, key="add_weight")
    
    if st.button("🦟 Add Transmission Path"):
        # Check if edge already exists
        edge_exists = False
        for u, v, w, l in st.session_state.edges_data:
            if (u == new_u and v == new_v) or (u == new_v and v == new_u):
                edge_exists = True
                break
        
        if edge_exists:
            st.error("🦟 Transmission path already exists between these palms!")
        elif new_u == new_v:
            st.error("Cannot connect a palm to itself!")
        else:
            # Generate new edge label
            next_num = len(st.session_state.edges_data) + 1
            new_label = f"Transmission {next_num}"
            
            st.session_state.edges_data.append((new_u, new_v, new_weight, new_label))
            st.session_state.graph = create_graph_from_edges(st.session_state.edges_data)
            st.success(f"✅ {new_label} created successfully!")
            st.rerun()

with tab_remove:
    st.write("**�️ Remove a transmission path:**")
    
    if len(st.session_state.edges_data) == 0:
        st.info("No transmission paths to remove!")
    else:
        edges_labels = [f"{label}: {u} ↔ {v} ({w}%)" for u, v, w, label in st.session_state.edges_data]
        selected_remove = st.selectbox("Select transmission path to remove:", edges_labels, key="remove_edge")
        
        if st.button("🗑️ Remove Transmission Path"):
            idx = edges_labels.index(selected_remove)
            removed = st.session_state.edges_data.pop(idx)
            st.session_state.graph = create_graph_from_edges(st.session_state.edges_data)
            st.success(f"✅ {removed[3]} removed!")
            st.rerun()

# Reset to default button
if st.sidebar.button("🔄 Reset to Default Farm"):
    st.session_state.edges_data = get_default_edges()
    st.session_state.vertices = ['Palm A', 'Palm B', 'Palm C', 'Palm D', 'Palm E']
    st.session_state.vertex_positions = {
        'Palm A': (0.5, 0.8),
        'Palm B': (0.25, 0.5),
        'Palm C': (0.75, 0.5),
        'Palm D': (0.33, 0.15),
        'Palm E': (0.67, 0.15)
    }
    st.session_state.graph = create_graph_from_edges(st.session_state.edges_data)
    st.rerun()

# Layout with two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🦟 RPW Spread Risk Network")
    fig = draw_graph(G, selected_vertex)
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)

with col2:
    st.subheader("🔗 Transmission Paths")
    st.write("**All RPW transmission paths with infection risk %:**")
    
    edge_data = []
    for u, v, w, label in st.session_state.edges_data:
        edge_data.append({
            "Transmission": label,
            "Between": f"{u} ↔ {v}",
            "Risk %": w
        })
    
    if edge_data:
        st.table(edge_data)
    else:
        st.info("No transmission paths in the network yet.")

# --- Path Analysis Section ---
st.markdown("---")
st.subheader(f"🦟 Most Likely RPW Spread Paths from {selected_vertex}")

# Find all paths from selected vertex
all_paths = find_all_paths_from_vertex(G, selected_vertex)

if not all_paths:
    st.warning(f"🦟 No spread paths found from {selected_vertex}")
else:
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["All Spread Paths 🛤️", "Highest Risk Paths 🔴", "Detailed Analysis 📋"])
    
    with tab1:
        st.write(f"**🦟 Total spread paths found: {len(all_paths)}**")
        st.info("Paths are ranked by cumulative infection risk (higher risk = longer path)")
        
        # Display all paths in a table format
        all_paths_data = []
        for i, path_info in enumerate(all_paths, 1):
            path_str = " → ".join(path_info['path'])
            all_paths_data.append({
                "Rank": i,
                "Source": path_info['start'],
                "Target": path_info['end'],
                "Path": path_str,
                "Cumulative Risk %": path_info['total_weight']
            })
        
        st.dataframe(all_paths_data, use_container_width=True, hide_index=True)
    
    with tab2:
        st.write("**🔴 Top 3 Highest Risk Spread Paths:**")
        st.warning("⚠️ These are the MOST LIKELY infection corridors to monitor closely!")
        
        top_3 = all_paths[:3]  # Already sorted by weight descending
        
        # Display top 3 using columns
        for i, path_info in enumerate(top_3, 1):
            col = st.container()
            with col:
                st.metric(
                    label=f"🔴 #{i} Risk Level: {path_info['start']} → {path_info['end']}",
                    value=f"{path_info['total_weight']}%",
                    delta="Cumulative Risk"
                )
                
                path_str = " → ".join(path_info['path'])
                st.write(f"**Infection Path:** {path_str}")
                st.write(f"**Transmission Chain:** " + " → ".join(path_info['edge_sequence']))
                st.divider()
    
    with tab3:
        # Expandable detailed view of all paths
        st.write("**Detailed Spread Path Analysis:**")
        for i, path_info in enumerate(all_paths, 1):
            with st.expander(f"📊 Path {i}: {path_info['start']} → {path_info['end']} (Risk: {path_info['total_weight']}%)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Infection route:**")
                    st.code(" → ".join(path_info['path']), language="text")
                
                with col2:
                    st.write("**Individual transmission risks:**")
                    for edge in path_info['edge_sequence']:
                        st.write(f"• {edge}")
                
                st.write(f"**Total Cumulative Infection Risk:** {path_info['total_weight']}%")

# --- Summary Statistics ---
st.markdown("---")
st.subheader("🦟 RPW Farm Risk Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Palms in Farm", len(vertices))

with col2:
    st.metric("Total Transmission Paths", G.number_of_edges())

with col3:
    st.metric("Potential Spread Paths", len(all_paths) if all_paths else 0)

# --- RPW Network Info Section ---
st.markdown("---")
st.subheader("ℹ️ Farm Network Information")

info_col1, info_col2 = st.columns(2)

with info_col1:
    palms_str = ", ".join(st.session_state.vertices)
    st.write(f"**🌴 Farm Palms:** {palms_str}")
    
with info_col2:
    st.write("**🦟 RPW Transmission Paths:**")
    edges_text = ""
    for u, v, w, label in st.session_state.edges_data:
        edges_text += f"- {label}: {u} → {v} (Risk: {w}%)\n"
    st.write(edges_text if edges_text else "No transmission paths defined")
    st.write(edges_text if edges_text else "No edges defined")

# Footer
st.markdown("---")
st.caption("🦟 Red Palm Weevil Farm Risk Simulator | Identify and Monitor High-Risk Spread Corridors | Powered by NetworkX & Streamlit")

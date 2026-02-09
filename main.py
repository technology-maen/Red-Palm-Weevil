"""
Streamlit GUI integrating sound detection, image-based hole detection, and RPW farm spread simulator.

Usage: streamlit run main.py

This app:
- accepts an optional audio file and runs `sound.detect_sound` if a trained model exists
- accepts an optional image and runs `hole.detect_holes_adaptive` to find holes
- computes a final risk score using `analyzer.compute_final_score`
- integrates RPW farm spread network to track infection corridors
- allows adding detected palms to the farm network with infection risk tracking
"""

import os
import tempfile
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from sound import detect_sound
from hole import detect_holes_adaptive, analyze_hole, print_results
from analyzer import compute_final_score


def save_upload_to_temp(uploaded, suffix="") -> Optional[str]:
    if uploaded is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(uploaded.getvalue())
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


def img_to_bytes_bgr(img_bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode('.jpg', img_bgr)
    return buf.tobytes()


def calculate_hole_distances(holes):
    """
    Calculate distances between all pairs oبكرا هيكون فيه ايڤينت يا شباب من اول الساعه ٧ وهيبقي صعب نعمل التمرين عشان المكان هيكون زحمة شوفو لو حابين نعمله بدري يا اما يوم تانيf holes and return sorted by distance.
    holes: list of dicts containing 'center' (x,y)
    Returns list of dicts with 'hole1','hole2','distance'
    """
    if len(holes) < 2:
        return []

    distances = []
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            c1 = holes[i]['center']
            c2 = holes[j]['center']
            distance = np.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)
            distances.append({'hole1': i + 1, 'hole2': j + 1, 'distance': float(distance)})

    distances.sort(key=lambda x: x['distance'])
    return distances


def get_distance_statistics(distances):
    """
    Return min, max, mean, median, std, count for distances list
    """
    if not distances:
        return None
    vals = [d['distance'] for d in distances]
    stats = {
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'mean': float(np.mean(vals)),
        'median': float(np.median(vals)),
        'std': float(np.std(vals)),
        'count': len(vals),
    }
    return stats


# ===== RPW FARM NETWORK FUNCTIONS =====

def initialize_farm_network():
    """Initialize the RPW farm network in session state"""
    if 'farm_palms' not in st.session_state:
        st.session_state.farm_palms = []
        st.session_state.farm_edges = []
        st.session_state.farm_graph = None
    if 'farm_positions' not in st.session_state:
        st.session_state.farm_positions = {}


def create_farm_graph():
    """Create NetworkX graph from farm data"""
    G = nx.Graph()
    if st.session_state.farm_palms:
        G.add_nodes_from(st.session_state.farm_palms)
        for u, v, risk, label in st.session_state.farm_edges:
            G.add_edge(u, v, weight=risk, label=label)
    st.session_state.farm_graph = G
    return G


def draw_farm_network():
    """Draw the RPW spread network visualization"""
    G = st.session_state.farm_graph
    
    if G is None or G.number_of_nodes() == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use spring layout if no positions set
    if not st.session_state.farm_positions:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = st.session_state.farm_positions
    
    # Draw nodes with red/orange theme for infection
    nx.draw_networkx_nodes(G, pos, node_color='#ff6b6b', node_size=1200,
                           edgecolors='#c92a2a', linewidths=3, ax=ax)
    
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold',
                            font_color='white', ax=ax)
    
    # Draw edges with color intensity based on infection percentage
    nx.draw_networkx_edges(G, pos, width=2.5, alpha=0.7, 
                          edge_color='#d63031', ax=ax)
    
    # Create edge labels with transmission risk percentages
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        risk = data.get('weight', 0)
        edge_labels[(u, v)] = f"{risk}%"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8,
                                 bbox=dict(boxstyle='round,pad=0.2',
                                          facecolor='#ffe0e0', alpha=0.9))
    
    ax.set_title('🦟 Farm RPW Spread Risk Network', fontsize=14, fontweight='bold', color='#c92a2a')
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def main():
    st.set_page_config(page_title="🦟 Red Palm Weevil Control Center", layout="wide")
    st.title("🦟 Red Palm Weevil — Control Center")
    
    # Initialize farm network
    initialize_farm_network()

    st.markdown("Complete palm tree assessments and build your farm's RPW spread monitoring network.")

    # Create main tabs
    tab_assessment, tab_farm_network = st.tabs(["🔍 Palm Assessment", "🦟 Farm Network & Spread Analysis"])
    
    # ===== TAB 1: ASSESSMENT =====
    with tab_assessment:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.header("Image (optional)")
            image_file = st.file_uploader("Upload trunk/crown image", type=["png", "jpg", "jpeg"], key="img")
            image_method = st.selectbox("Detection method", options=["combined", "dark", "texture"], index=0)
            show_debug = st.checkbox("Show debug mask/image", value=False)

            st.write("---")
            st.header("Manual hole/trunk observations (overrides/augments image)")
            trunk_strength = st.slider("Trunk strength (1 weak - 10 strong)", 0.0, 10.0, 5.0, 0.5)
            hole_depth_manual = st.number_input("Hole depth (cm) — manual (0 = unknown)", min_value=0.0, value=0.0, step=1.0)
            hole_wet = st.checkbox("Hole area wet / moist", value=False)
            exposed_crown = st.checkbox("Exposed crown present", value=False)

        with col_right:
            st.header("Audio (optional)")
            audio_file = st.file_uploader("Upload short audio (wav/mp3)", type=["wav", "mp3"], key="audio")
            st.write("---")
            st.header("Biases")
            age = st.number_input("Age of palm (years)", min_value=0.0, value=10.0, step=1.0)
            humidity = st.slider("Local humidity (%)", 0, 100, 50)

            st.write("---")
            assess = st.button("Assess")

        # default outputs
        annotated_img_bytes = None
        detected_holes = []
        combined_mask = None
        debug_img = None
        sound_detected = None
        sound_msg = "No audio provided or model missing"

        if assess:
            # handle image
            img_path = save_upload_to_temp(image_file, suffix=".jpg") if image_file is not None else None
            if img_path:
                try:
                    result_img, holes, mask, dbg = detect_holes_adaptive(img_path, method=image_method)
                    detected_holes = holes
                    combined_mask = mask
                    debug_img = dbg
                    annotated_img_bytes = img_to_bytes_bgr(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                except Exception as e:
                    st.error(f"Image processing error: {e}")
                finally:
                    try:
                        os.remove(img_path)
                    except Exception:
                        pass

        # handle audio
        audio_path = save_upload_to_temp(audio_file, suffix=".wav") if audio_file is not None else None
        if audio_path:
            try:
                detected = detect_sound(audio_path)
                sound_detected = bool(detected)
                sound_msg = "Detected: TRUE (similar to training sounds)" if sound_detected else "Detected: FALSE"
            except Exception as e:
                sound_detected = False
                sound_msg = f"Sound detector error: {e}"
            finally:
                try:
                    os.remove(audio_path)
                except Exception:
                    pass

        # derive hole_depth from image if not provided manually
        hole_depth_cm = float(hole_depth_manual)
        if hole_depth_cm <= 0 and detected_holes:
            # heuristic mapping: based on number of holes and largest area ratio
            num = len(detected_holes)
            areas = [h['area'] for h in detected_holes]
            largest = max(areas) if areas else 0
            # area ratio relative to image
            # Try to estimate approximate depth: more/larger holes => deeper
            img_area = result_img.shape[0] * result_img.shape[1]
            ratio = largest / img_area if img_area > 0 else 0
            # map ratio to cm (0..30 cm)
            estimated_cm = min(30.0, max(0.0, (ratio * 1000) + (num * 2)))
            hole_depth_cm = round(estimated_cm, 1)

        # Use manual wet/exposed unless image analysis suggests otherwise (not automated yet)

        # run hole analyzer
        hole_res = analyze_hole(trunk_strength=trunk_strength, hole_depth_cm=hole_depth_cm, wet=hole_wet, exposed_crown=exposed_crown)

        # make sure sound_detected is boolean
        if sound_detected is None:
            sound_detected = False

        final = compute_final_score(sound_bool=sound_detected, hole_result=hole_res,
                                    age_years=age, humidity_pct=float(humidity),
                                    trunk_strength_input=trunk_strength,
                                    exposed_crown_input=exposed_crown, wet_input=hole_wet)

        # Show results
        st.subheader("Assessment result")
        st.metric("Risk score", f"{final['score']*100:.1f}%")
        st.write(f"Category: **{final['category']}**")

        # Add to Farm button
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            palm_name_to_add = st.text_input("Palm identifier for farm:", value=f"Palm_{len(st.session_state.farm_palms) + 1}", key="palm_id_input")
        
        with col_btn2:
            if st.button("🌴 Add Detected Palm to Farm"):
                if palm_name_to_add:
                    # Calculate infection risk based on assessment
                    infection_risk = int(final['score'] * 100)
                    
                    if palm_name_to_add not in st.session_state.farm_palms:
                        st.session_state.farm_palms.append(palm_name_to_add)
                        if not st.session_state.farm_positions:
                            st.session_state.farm_positions = {}
                        # Add automatic position
                        angle = len(st.session_state.farm_palms) * 2 * np.pi / 5
                        st.session_state.farm_positions[palm_name_to_add] = (0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle))
                        create_farm_graph()
                        st.success(f"✅ {palm_name_to_add} added to farm with {infection_risk}% infection risk!")
                        st.balloons()
                    else:
                        st.warning(f"🌴 {palm_name_to_add} already in farm")

        st.write("### Breakdown")
        st.json(final['breakdown'])

        st.write("### Sound detector")
        st.write(sound_msg)

        st.write("### Hole analysis (heuristic)")
        st.json(hole_res)

        st.write("### Image detection summary")
        st.write(f"Holes found: {len(detected_holes)}")
        if len(detected_holes) > 0:
            st.write(detected_holes)

            # Calculate pairwise distances and show statistics
            distances = calculate_hole_distances(detected_holes)
            stats = get_distance_statistics(distances)
            if distances:
                st.write("#### Distances (pixels) — sorted by proximity")
                st.table([{ 'Priority': i+1, 'Holes': f"H{d['hole1']} ↔ H{d['hole2']}", 'Distance(px)': round(d['distance'],2)} for i,d in enumerate(distances)])

            if stats:
                st.write("#### Distance statistics")
                st.write(f"Pairs: {stats['count']}, min: {stats['min']:.2f}px, max: {stats['max']:.2f}px, mean: {stats['mean']:.2f}px, median: {stats['median']:.2f}px, std: {stats['std']:.2f}px")

        if annotated_img_bytes is not None:
            st.write("### Annotated image")
            st.image(annotated_img_bytes, use_column_width=True)
            if show_debug and debug_img is not None:
                st.write("### Debug image / mask")
                # show mask
                st.image(img_to_bytes_bgr(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)), caption="debug (RGB)")
                if combined_mask is not None:
                    _, mask_buf = cv2.imencode('.jpg', combined_mask)
                    st.image(mask_buf.tobytes(), caption="mask")

        else:
            st.info("Set inputs and press Assess to run the estimator.")
    
    # ===== TAB 2: FARM NETWORK =====
    with tab_farm_network:
        st.header("🦟 Farm RPW Spread Network")
        st.markdown("""
        **Monitor and manage RPW spread across your farm.** 
        After assessments, add detected palms to build your farm network. 
        The app identifies the highest-risk infection corridors based on transmission percentages.
        """)
        
        # Sidebar for farm management
        col_farm_left, col_farm_right = st.columns([2, 1])
        
        with col_farm_left:
            st.subheader("📊 Farm Network Visualization")
            
            if st.session_state.farm_graph and st.session_state.farm_graph.number_of_nodes() > 0:
                fig = draw_farm_network()
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Add a checkbox to specify starting point for spread analysis
                analyze_spread = st.checkbox("Analyze spread patterns", value=False)
                
                if analyze_spread and st.session_state.farm_graph.number_of_edges() > 0:
                    st.subheader("🦟 Spread Analysis")
                    source_palm = st.selectbox("Select infection source palm:", st.session_state.farm_palms)
                    
                    # Find all paths using DFS
                    all_paths = []
                    G = st.session_state.farm_graph
                    
                    for target in st.session_state.farm_palms:
                        if source_palm != target:
                            try:
                                paths = list(nx.all_simple_paths(G, source_palm, target))
                                for path in paths:
                                    total_risk = 0
                                    edge_info = []
                                    for i in range(len(path) - 1):
                                        u, v = path[i], path[i + 1]
                                        risk = G[u][v]['weight']
                                        total_risk += risk
                                        edge_info.append(f"{u}→{v}({risk}%)")
                                    all_paths.append({
                                        'source': source_palm,
                                        'target': target,
                                        'path': path,
                                        'risk': total_risk,
                                        'edges': edge_info
                                    })
                            except nx.NetworkXNoPath:
                                pass
                    
                    if all_paths:
                        # Sort by risk (highest first)
                        all_paths.sort(key=lambda x: x['risk'], reverse=True)
                        
                        st.warning(f"⚠️ **Top 3 Highest Risk Spread Corridors from {source_palm}:**")
                        
                        for i, path_info in enumerate(all_paths[:3], 1):
                            col = st.container()
                            st.metric(
                                label=f"🔴 Risk #{i}: {path_info['source']} → {path_info['target']}",
                                value=f"{path_info['risk']}%",
                                delta="Cumulative Risk"
                            )
                            st.write(f"**Path:** {' → '.join(path_info['path'])}")
                            st.write(f"**Transmissions:** {', '.join(path_info['edges'])}")
                            st.divider()
            else:
                st.info("📍 No palms added to farm yet. Add detected palms from assessments to build the network.")
        
        with col_farm_right:
            st.subheader("🌴 Farm Palms")
            
            # Show current farm palms
            if st.session_state.farm_palms:
                st.write(f"**Palms in farm:** {len(st.session_state.farm_palms)}")
                for palm in st.session_state.farm_palms:
                    st.write(f"• {palm}")
            else:
                st.write("No palms added yet")
            
            st.markdown("---")
            st.subheader("🦟 Add Palm")
            
            # Quick add form
            col_form1, col_form2 = st.columns(2)
            with col_form1:
                new_palm_name = st.text_input("Palm identifier:", key="new_palm_form")
            with col_form2:
                new_palm_risk = st.number_input("Initial infection risk (%):", min_value=0, max_value=100, value=0, key="new_risk_form")
            
            if st.button("➕ Add Palm to Farm"):
                if new_palm_name and new_palm_name not in st.session_state.farm_palms:
                    st.session_state.farm_palms.append(new_palm_name)
                    if not st.session_state.farm_positions:
                        st.session_state.farm_positions = {}
                    # Add automatic position
                    angle = len(st.session_state.farm_palms) * 2 * np.pi / 5
                    st.session_state.farm_positions[new_palm_name] = (0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle))
                    create_farm_graph()
                    st.success(f"✅ {new_palm_name} added!")
                    st.rerun()
                elif new_palm_name in st.session_state.farm_palms:
                    st.warning(f"🌴 {new_palm_name} already exists")
            
            st.markdown("---")
            st.subheader("🔗 Add Transmission")
            
            if len(st.session_state.farm_palms) >= 2:
                col_edge1, col_edge2, col_edge3 = st.columns(3)
                
                with col_edge1:
                    source = st.selectbox("From:", st.session_state.farm_palms, key="edge_from")
                
                with col_edge2:
                    target = st.selectbox("To:", st.session_state.farm_palms, key="edge_to")
                
                with col_edge3:
                    risk = st.number_input("Risk %:", min_value=0, max_value=100, value=50, key="edge_risk")
                
                if st.button("➕ Add Transmission Path"):
                    if source != target:
                        # Check if edge exists
                        edge_exists = any((u == source and v == target) or (u == target and v == source) 
                                         for u, v, _, _ in st.session_state.farm_edges)
                        if not edge_exists:
                            label = f"T{len(st.session_state.farm_edges) + 1}"
                            st.session_state.farm_edges.append((source, target, risk, label))
                            create_farm_graph()
                            st.success(f"✅ Transmission path added!")
                            st.rerun()
                        else:
                            st.warning("Path already exists")
                    else:
                        st.error("Select different palms")
            
            # Show current edges
            if st.session_state.farm_edges:
                st.markdown("---")
                st.write("**Current Transmission Paths:**")
                for u, v, risk, label in st.session_state.farm_edges:
                    st.write(f"• {u} → {v}: {risk}%")


if __name__ == '__main__':
    main()

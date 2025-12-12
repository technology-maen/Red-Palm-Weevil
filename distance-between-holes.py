import numpy as np


def calculate_hole_distances(holes):
    """
    Calculate distances between all pairs of holes and return sorted by priority.

    Args:
        holes: List of hole dictionaries, each containing 'center' key with (x, y) coordinates

    Returns:
        distances: List of dictionaries with 'hole1', 'hole2', 'distance', sorted by distance
    """
    if len(holes) < 2:
        return []

    # Store all distances with hole pairs
    distances = []

    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            center1 = holes[i]['center']
            center2 = holes[j]['center']

            # Calculate Euclidean distance
            distance = np.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2)

            distances.append({
                'hole1': i + 1,
                'hole2': j + 1,
                'distance': distance
            })

    # Sort distances from smallest to largest (highest to lowest priority)
    distances.sort(key=lambda x: x['distance'])

    return distances


def print_distance_distribution(distances):
    """
    Print the distance distribution in a formatted table.

    Args:
        distances: List of distance dictionaries from calculate_hole_distances()
    """
    if not distances:
        print("No distances to display (less than 2 holes detected)")
        return

    print(f"\n{'=' * 60}")
    print(f"DISTANCES BETWEEN HOLES (Sorted by Priority)")
    print(f"{'=' * 60}")

    # Print table header
    print(f"\n{'Priority':<10} {'Holes':<15} {'Distance (pixels)':<20}")
    print("-" * 50)

    # Print each distance with priority
    for priority, dist_info in enumerate(distances, 1):
        hole_pair = f"H{dist_info['hole1']} ↔ H{dist_info['hole2']}"
        print(f"{priority:<10} {hole_pair:<15} {dist_info['distance']:.2f}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(
        f"CLOSEST PAIR: Hole {distances[0]['hole1']} and Hole {distances[0]['hole2']} ({distances[0]['distance']:.2f} pixels)")
    print(
        f"FARTHEST PAIR: Hole {distances[-1]['hole1']} and Hole {distances[-1]['hole2']} ({distances[-1]['distance']:.2f} pixels)")
    print(f"{'=' * 60}")


def get_distance_statistics(distances):
    """
    Calculate statistics about the distance distribution.

    Args:
        distances: List of distance dictionaries from calculate_hole_distances()

    Returns:
        stats: Dictionary containing min, max, mean, median distances
    """
    if not distances:
        return None

    distance_values = [d['distance'] for d in distances]

    stats = {
        'min': min(distance_values),
        'max': max(distance_values),
        'mean': np.mean(distance_values),
        'median': np.median(distance_values),
        'std': np.std(distance_values),
        'count': len(distance_values)
    }

    return stats


# Example usage
if __name__ == "__main__":
    # Example holes data
    example_holes = [
        {'center': (100, 150), 'area': 500},
        {'center': (200, 180), 'area': 600},
        {'center': (150, 300), 'area': 450},
        {'center': (350, 250), 'area': 700}
    ]

    # Calculate distances
    distances = calculate_hole_distances(example_holes)

    # Print distribution
    print_distance_distribution(distances)

    # Get statistics
    stats = get_distance_statistics(distances)
    if stats:
        print(f"\n{'=' * 60}")
        print("DISTANCE STATISTICS")
        print(f"{'=' * 60}")
        print(f"Number of pairs: {stats['count']}")
        print(f"Minimum distance: {stats['min']:.2f} pixels")
        print(f"Maximum distance: {stats['max']:.2f} pixels")
        print(f"Mean distance: {stats['mean']:.2f} pixels")
        print(f"Median distance: {stats['median']:.2f} pixels")
        print(f"Standard deviation: {stats['std']:.2f} pixels")
        print(f"{'=' * 60}")
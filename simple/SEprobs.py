#!/usr/bin/env python3
import random as rnd
import argparse
from scipy.stats import poisson_binom
from typing import Dict, Any, Tuple


def parse_arguments() -> Tuple[Dict[Any, float], argparse.Namespace]:
    """Parse command line arguments in format: <probability>-<name>"""
    parser = argparse.ArgumentParser(
        description='Calculate probabilities of selecting items with given availability probabilities.',
        epilog='Example: SEprobs.py 100-ItemA 20-ItemB 1-ItemC'
    )
    parser.add_argument(
        'items',
        nargs='+',
        metavar='<prob>-<name> or <name>',
        help='Items in format <probability>-<name> where probability is 0-100, or just <name> for 100%%'
    )
    parser.add_argument(
        '--analytic',
        action='store_true',
        default=False,
        help='Show analytical results (default if no method specified)'
    )
    parser.add_argument(
        '--empiric',
        action='store_true',
        default=False,
        help='Show empirical simulation results'
    )

    args = parser.parse_args()

    # If neither flag is specified, default to analytic
    if not args.analytic and not args.empiric:
        args.analytic = True

    base_probs = {}
    for arg in args.items:
        parts = arg.rsplit('-', 1)

        if len(parts) == 1:
            # No dash found, treat as 100% probability
            name = parts[0]
            prob = 1.0
        else:
            # Dash found, try to parse probability
            prob_str, name = parts
            try:
                prob = float(prob_str) / 100.0
                if prob < 0 or prob > 1:
                    parser.error(f"Probability must be between 0 and 100, got {prob_str}")
            except ValueError:
                # First part isn't a number, treat entire arg as name with 100% probability
                name = arg
                prob = 1.0

        base_probs[name] = prob

    return base_probs, args


base_probs, args = parse_arguments()

# Check if any item has 100% probability
has_guaranteed_item = any(prob >= 1.0 for prob in base_probs.values())



def empiric_unit():
    final_items = []
    for item, prob in base_probs.items():
        if rnd.random() < prob:
            final_items.append(item)

    if not final_items:
        return None  # No selection made
    return rnd.choice(final_items)

def empiric_bulk(n):
    results = {}
    for _ in range(n):
        res = empiric_unit()
        try :
            results[res] += 1
        except KeyError:
            results[res] = 1

    for item in results:
        results[item] /= n

    # Convert None key to "No selection" string for display
    if None in results:
        results["<NO SELECTION>"] = results.pop(None)

    return results



def analytic():
    results = {}
    for item, prob in base_probs.items():
        non_item = [p for i, p in base_probs.items() if i != item]

        results[item] = sum((1/(i + 1)) * poisson_binom.pmf(i, non_item) for i in range(len(base_probs))) * prob

    # If no item is guaranteed, calculate probability of no selection
    if not has_guaranteed_item:
        # Probability that all items are unavailable
        prob_no_selection = 1.0
        for prob in base_probs.values():
            prob_no_selection *= (1.0 - prob)
        results["<NO SELECTION>"] = prob_no_selection

    return results


def print_results(results, title):
    """Print results as a formatted table, sorted by probability (descending)"""
    print(f"\n{title}")
    print("-" * 50)

    # Sort by probability (descending)
    sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # Find max item name length for alignment
    max_name_len = max(len(str(item)) for item in results.keys())

    # Print header
    print(f"{'Item':<{max_name_len}}  {'Probability':>12}")
    print(f"{'-' * max_name_len}  {'-' * 12}")

    # Print each item
    for item, prob in sorted_items:
        percent = prob * 100
        print(f"{str(item):<{max_name_len}}  {percent:>11.2f}%")


# Run calculations and print results
if args.analytic:
    results = analytic()
    print_results(results, "Analytical Results")

if args.empiric:
    results = empiric_bulk(100000)
    print_results(results, "Empirical Results (100,000 simulations)")

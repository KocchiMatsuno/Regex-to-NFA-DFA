import itertools
import re
from collections import deque
from graphviz import Digraph

# State Class
class State:
    _counter = itertools.count()

    def __init__(self):
        self.name = f"q{next(State._counter)}"
        self.transitions = {}
        self.epsilon_transitions = []

    def __repr__(self):
        return self.name

    @classmethod
    def reset_counter(cls):
        cls._counter = itertools.count()

# NFA Class
class NFA:
    def __init__(self, start, end, final_states=None):
        self.start = start
        self.end = end
        self.final_states = final_states or []

    def visualize(self):
        """Visualize the NFA using Graphviz."""
        dot = Digraph(format='png', engine='dot')  # Use 'dot' layout engine for directed graphs
        visited = set()

        # Recursive function to traverse states and add transitions
        def traverse(state):
            if state in visited:
                return
            visited.add(state)

            # Add current state node
            color = "lightgreen" if state in self.final_states else "lightblue"
            dot.node(state.name, state.name, style="filled", fillcolor=color)

            # Add transitions
            for symbol, next_states in state.transitions.items():
                for next_state in next_states:
                    dot.edge(state.name, next_state.name, label=symbol)
                    traverse(next_state)

            # Add epsilon transitions
            for next_state in state.epsilon_transitions:
                dot.edge(state.name, next_state.name, label="ε")
                traverse(next_state)

        # Start traversal from the start state
        traverse(self.start)
        
        # Render the NFA graph
        dot.render('nfa_visualization', view=True)

# DFA Class
class DFA:
    def __init__(self, start, states, transitions, final_states):
        self.start = start
        self.states = states
        self.transitions = transitions
        self.final_states = final_states

    def visualize(self):
        """Visualize the DFA using Graphviz."""
        dot = Digraph(format='png', engine='dot')  # Use 'dot' layout engine for directed graphs

        # Add nodes (states)
        for state in self.states:
            color = "lightgreen" if state in self.final_states else "lightblue"
            dot.node(state.name, state.name, style="filled", fillcolor=color)

        # Add edges (transitions)
        for (state, symbol), next_state in self.transitions.items():
            dot.edge(state.name, next_state.name, label=symbol)

        # Render the graph to a file (DFA visualization)
        dot.render('dfa_visualization', view=True)

# Utility Functions
def epsilon_closure(states):
    stack = list(states)
    closure = set(states)

    while stack:
        state = stack.pop()
        for next_state in state.epsilon_transitions:
            if next_state not in closure:
                closure.add(next_state)
                stack.append(next_state)

    return closure

# NFA to DFA conversion
def nfa_to_dfa(nfa):
    State.reset_counter()

    # Step 1: Start with epsilon closure for the NFA's start state
    start_closure = frozenset(epsilon_closure({nfa.start}))

    # States, transitions, and final states initialization
    states = {start_closure}
    transitions = {}
    final_states = set()
    worklist = deque([start_closure])

    # Mapping closures to DFA states
    state_mapping = {start_closure: State()}
    alphabet = set()

    # Debugging: Print initial epsilon closure
    print(f"Initial epsilon closure: {start_closure}")

    while worklist:
        current_closure = worklist.popleft()
        current_state = state_mapping[current_closure]

        # Mark this closure as a final state if it contains any of the NFA's final states
        if any(state in nfa.final_states for state in current_closure):
            final_states.add(current_state)

        # Collect the alphabet from the transitions of all states in the current closure
        symbols = {symbol for state in current_closure for symbol in state.transitions.keys()}
        alphabet.update(symbols)

        for symbol in alphabet:
            next_closure = frozenset(
                s
                for state in current_closure
                for next_state in state.transitions.get(symbol, [])
                for s in epsilon_closure({next_state})
            )

            if next_closure and next_closure not in states:
                states.add(next_closure)
                worklist.append(next_closure)
                state_mapping[next_closure] = State()

            if next_closure:
                next_state = state_mapping[next_closure]
                transitions[(current_state, symbol)] = next_state

    return DFA(state_mapping[start_closure], set(state_mapping.values()), transitions, final_states)

# Regex-related functions
def tokenize_regex(regex):
    return re.findall(r'[a-zA-Z0-9]|[()|*.]', regex)

def insert_concatenation_operators(regex):
    result = []
    for i, char in enumerate(regex):
        result.append(char)
        if (char.isalnum() or char in ')*') and i + 1 < len(regex) and (regex[i + 1].isalnum() or regex[i + 1] == '('):
            result.append('.')
    return ''.join(result)

def precedence(op):
    return {'*': 3, '.': 2, '|': 1}.get(op, 0)

def to_postfix(tokens):
    output = []
    operators = []

    for token in tokens:
        if token.isalnum():
            output.append(token)
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()
        else:
            while operators and precedence(operators[-1]) >= precedence(token):
                output.append(operators.pop())
            operators.append(token)

    while operators:
        output.append(operators.pop())

    return output

# Step 4 Enhancements to regex_to_nfa
def regex_to_nfa(regex):
    State.reset_counter()
    regex = insert_concatenation_operators(regex)
    tokens = tokenize_regex(regex)
    postfix = to_postfix(tokens)
    stack = []

    def create_basic_nfa(symbol):
        start = State()
        end = State()
        start.transitions[symbol] = [end]
        return NFA(start, end, [end])

    for token in postfix:
        if token.isalnum():
            stack.append(create_basic_nfa(token))

        # Kleene Star
        elif token == '*':
            nfa = stack.pop()
            start = State()
            end = State()
            start.epsilon_transitions.append(nfa.start)  # New start connects to old start
            start.epsilon_transitions.append(end)        # New start connects to new end
            nfa.end.epsilon_transitions.append(nfa.start)  # Loop back to old start
            nfa.end.epsilon_transitions.append(end)        # Old end connects to new end
            stack.append(NFA(start, end, [end]))

        # Union 
        elif token == '|':
            nfa2 = stack.pop()
            nfa1 = stack.pop()
            start = State()
            end = State()
            start.epsilon_transitions.extend([nfa1.start, nfa2.start])
            nfa1.end.epsilon_transitions.append(end)
            nfa2.end.epsilon_transitions.append(end)
            stack.append(NFA(start, end, [end]))

        # Concatenation Fix
        elif token == '.':
            nfa2 = stack.pop()
            nfa1 = stack.pop()
            nfa1.end.epsilon_transitions.append(nfa2.start)
            stack.append(NFA(nfa1.start, nfa2.end, [nfa2.end]))

    return stack.pop()

# Debug Function to Print NFA States and Transitions
def debug_nfa(nfa):
    print("\n--- NFA Debug Output ---")
    visited = set()

    def traverse(state):
        if state in visited:
            return
        visited.add(state)
        print(f"State: {state}")
        for symbol, next_states in state.transitions.items():
            for next_state in next_states:
                print(f"  {state} --{symbol}--> {next_state}")
        for next_state in state.epsilon_transitions:
            print(f"  {state} --ε--> {next_state}")
            traverse(next_state)

    traverse(nfa.start)
    print("--- End of Debug ---\n")

# Example Usage
regex_input = input("Enter a Regular Expression: ")
nfa_instance = regex_to_nfa(regex_input)

# Debug the NFA structure before visualization
debug_nfa(nfa_instance)

# Visualize the NFA
nfa_instance.visualize()

# Convert to DFA and visualize it
dfa_instance = nfa_to_dfa(nfa_instance)
dfa_instance.visualize()

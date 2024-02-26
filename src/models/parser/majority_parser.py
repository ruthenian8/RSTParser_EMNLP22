from .shift_reduce_parser_v1 import ShiftReduceParserV1
from data.doc import Doc
from data.tree import RSTTree


class MajorityParser(ShiftReduceParserV1):
    @staticmethod
    def tree_to_array(tree):
        # Initialize a list with the root node
        array = [None] * 1024  # Start with a large size to accommodate all nodes, will trim excess None at the end
        queue = [(tree, 0)]  # Queue of (node, index) tuples for BFS
        
        while queue:
            node, index = queue.pop(0)  # Get the current node and its index in the array
            
            if isinstance(node, RSTTree):
                if node.label() is not None:
                    array[index] = node.label()  # Set the label for non-leaf nodes
                # Enqueue children
                if len(node) >= 1 and node[0]:
                    queue.append((node[0], 2 * index + 1))
                if len(node) == 2 and node[1]:
                    queue.append((node[1], 2 * index + 2))
            else:
                array[index] = node  # Set the value directly for leaf nodes
        
        # Trim the excess None values from the array
        last_index = max(i for i, x in enumerate(array) if x is not None)
        return array[:last_index + 1]
    
    def parse(self, doc: Doc):
        bert_output = self.classifier.encoder(doc)
        n_edus = len(doc.edus)

        state = ShiftReduceState(n_edus)
        while not state.is_end():
            s1, s2, q1 = state.get_state()
            span = {"s1": s1, "s2": s2, "q1": q1}
            feat = {"org": self.get_organization_features(s1, s2, q1, doc, self.classifier.device)}

            # predict action and labels
            act, nuc, rel = self.select_action_and_labels(bert_output, span, feat, state)

            # update stack and queue
            state.operate(act, nuc, rel)

        tree = state.get_tree()
        return tree
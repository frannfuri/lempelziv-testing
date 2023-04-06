import numpy as np

def KC_complexity_lempelziv_count(sequence, normalize=True, permutation=False, dim=2):
    ''' Computes LZC counts from symbolic sequences'''
    # Initialize variables
    n = len(sequence)
    u, v, w = 0, 1, 1
    v_max = 1 # Stores the length of the longest pattern in the "look-ahead" (LA) that has been
    # matched somewhere in the "search buffer" (SB).
    complexity = 1

    while True:
        if sequence[u + v - 1] == sequence[w + v - 1]: # Increase "v" as long as the pattern matches, i.e. as long as
            # sequence[w+v-1] bit string can be reconstructed by sequence[u+v-1] bit string. Note that the matched
            # pattern can "run over" "w" because the pattern starts copying itself (see LZ 76 paper).
            v += 1
            if w + v >= n: # If reach the end of the string while matching, then need to add that to the tokens, and stop.
                complexity += 1
                break
        else:
            if v > v_max:
                v_max = v
            u += 1 # Increase "u" while the bit doesn't match, looking for a previous
            # occurrence of a pattern. sequence[u+v-1] is scanning the SB.
            if u == w: # Stop looking when "u" catches up with the first bit of the LA part.
                complexity += 1
                w += v_max # Move the beginning of the LA to the end of the newly matched pattern.
                if w >= n: # Stop if LA surpasses length of string.
                    break
                else: # After step.
                    u = 0 # Reset searching index to beginning of SB (beginning of string).
                    v = 1 # Reset pattern matching index.
                    v_max = 1 # Reset max length of matched pattern to k.
            else:
                v = 1 # Finished matching a pattern in the SB, and we reset the matched pattern length counter.

    if normalize is True:
        if permutation is False:
            out = (complexity * np.log2(n)) / n
        else:
            out = (complexity * np.log(n) / np.log(np.math.factorial(dim))) / n
    else:
        # TODO
        out = 0
        raise NotImplementedError('The non-normalized option hasnt be implemented yet.')
    return out
#!/usr/bin/env python

# Generic
import logging
from collections import Counter

# Libs
import itertools

##################
# Configurations #
##################


#################################
# Alignment Helper Class - Cell #
#################################

class Cell:
    """ Helper class to track traceback operations during
        pairwise alignment. Traceback is configured as a
        linked list of Cell objects.
        
    Attributes:
        __prev (Cell): Previous cell leading to this cell
        
        e1 (str): Aligned element on header sequence m
        e2 (str): Aligned element on header sequence n
        score (float): Score accrued during propagation
    """

    def __init__(self):
        self.__prev = None
        self.e1 = None
        self.e2 = None
        self.score = 0
        
    def __repr__(self):
        return f"[({self.e1},{self.e2}), {self.score}]"
        
    @property
    def prev(self):
        return self.__prev
    
    @prev.setter
    def prev(self, cell):
        self.__prev = cell

#################################################
# Data Alignment Class - PairwiseFeatureAligner #
#################################################

class PairwiseFeatureAligner:
    """ Aligns 2 header sequences by filling up unaligned indexes with gaps.
        There are 2 kinds of alignments that can be done; global alignment
        is done using the Needleman-wunsch algorithm, while local alignments
        are obtained using the Smith-waterman algorithm.
        
    Args:
        m_headers (list(str)): Headers for dataset A
        n_headers (list(str)): Headers for dataset B
    
    Attributes:
        __grid (list(list(float))): DP matrix storing tracebacks
        
        m_headers (list(str)): Headers for dataset A
        n_headers (list(str)): Headers for dataset B
    """
    
    def __init__(self, m_headers, n_headers):
        
        self.__grid = [
            [Cell() 
             for j in range(len(n_headers)+1)
            ]for i in range(len(m_headers)+1)
        ]
        
        self.m_headers = m_headers
        self.n_headers = n_headers

    ###########
    # Getters #
    ###########
    
    @property
    def grid(self):
        [print(row) for row in self.__grid]
        
    def get_cell(self, i, j):
        return self.__grid[i][j]
    
    ###########
    # Helpers #
    ###########
    
    def resolve_optimal_path(self, i, j, match, mismatch, gap, algo):
        """ Automates & resolves the optimal path by maximising the 
            score for a specified cell at the i-th row & j-th column.
            Route scores are calculated based on backward adjacent cell 
            values, and are penalised based on a set of specified
            match-mismatch & indel penalties. The optimal score is then
            computed based on the selected algorithm, after which the 
            traceback & optimal alignment is logged in the current cell.
            
        Args:
            i (int): Row index of target cell
            j (int): Column index of target cell
            match (float): Reward earned for a correct elementwise match
            mismatch (float): Penalty incurred for a wrong elementwise match
            gap (float): Penalty incurred to open a gap
            algo (str): Algorithm to use for pairwise alignment 
                        Supported algorithms: 1) Global (Needleman-wunsch)
                                              2) Local  (Smith-waterman)
        Returns:
            Aligned cell (Cell)
        """
        curr_cell = self.get_cell(i, j)
        curr_m_feature = self.m_headers[i-1]
        curr_n_feature = self.n_headers[j-1]
        
        # Calculate score if gap applied on n
        adjacent_cell = self.get_cell(i, j-1)
        adj_score = adjacent_cell.score + gap

        # Calculate score if match/mismatch
        diagonal_cell = self.get_cell(i-1, j-1)
        if curr_m_feature == curr_n_feature:
            diag_score = diagonal_cell.score + match
        else:
            diag_score = diagonal_cell.score + mismatch

        # Calculate score if gap applied on m
        top_cell = self.get_cell(i-1, j)
        top_score = top_cell.score + gap

        # Find maximum score
        if algo =='global':
            optimal_score = max(adj_score, diag_score, top_score)
        elif algo == 'local':
            optimal_score = max(0, adj_score, diag_score, top_score)
        else:
            raise NotImplementedError("Algorithm not supported!")

        # Update the current cell's score with max score
        curr_cell.score = optimal_score

        # Cache adjacent path if it was optimal
        if adj_score == optimal_score:
            curr_cell.prev = self.get_cell(i, j-1)
            curr_cell.e2 = curr_n_feature 

        # Cache diagonal path if it was optimal
        elif diag_score == optimal_score:
            curr_cell.prev = self.get_cell(i-1, j-1)
            curr_cell.e1 = curr_m_feature
            curr_cell.e2 = curr_n_feature

        # Cache top path if it was optimal
        else:
            curr_cell.prev = self.get_cell(i-1, j)
            curr_cell.e1 = curr_m_feature
            
        return curr_cell
    
    
    def perform_global_alignment(self, match, mismatch, gap):
        """ Execute the Needleman-Wunsch algorithm on specified headers
            to capture global alignment in the dynamic programming grid,
            given a set of specified match-mismatch & indel penalties.
            
        Args:
            match (float): Reward earned for a correct elementwise match
            mismatch (float): Penalty incurred for a wrong elementwise match
            gap (float): Penalty incurred to open a gap
        Returns:
            Configured DP grid (list(list(Cell)))
        """
        for i, row in enumerate(self.__grid):
            
            for j, col in enumerate(row):
                
                curr_cell = self.get_cell(i, j)
                curr_m_feature = self.m_headers[i-1]
                curr_n_feature = self.n_headers[j-1]
                
                # Skip the first cell
                if i == 0 and j == 0:
                    continue
                
                # Initialise 1st row with gap penalties
                elif i == 0 and j > 0:
                    curr_cell.score = self.get_cell(i, j-1).score + gap
                    curr_cell.e2 = curr_n_feature
                    curr_cell.prev = self.get_cell(i, j-1)
                
                # Initialise 1st column with gap penalties
                elif i > 0 and j == 0:
                    curr_cell.score = self.get_cell(i-1, j).score + gap
                    curr_cell.e1 = curr_m_feature
                    curr_cell.prev = self.get_cell(i-1, j)
                    
                else:
                    self.resolve_optimal_path(i, j, match, mismatch, gap, 'global')
        
        return self.__grid
    
    
    def perform_local_alignment(self, match, mismatch, gap):
        """ Execute the Smith-Waterman algorithm on specified headers
            to capture local alignments in the dynamic programming grid,
            given a set of specified match-mismatch & indel penalties.
            
        Args:
            match (float): Reward earned for a correct elementwise match
            mismatch (float): Penalty incurred for a wrong elementwise match
            gap (float): Penalty incurred to open a gap
        Returns:
            Configured DP grid (list(list(Cell)))
        """
        for i, row in enumerate(self.__grid):
            
            for j, col in enumerate(row):
                
                # Local pairwise alignment dictates that the 1st row & column
                # of the DP grid be initialised with the value of 0. For these
                # cells, traceback is unnecessary, since local alignment
                # terminates at any cell with the score of 0 to capture 
                # earlystoppping. Since in the default score initialised
                # in helper class Cell is already 0, there is no need to
                # further operate on these cells. Only start traceback logging
                # for 2nd row & columns onwards.
                if i != 0 and j != 0:
                    self.resolve_optimal_path(i, j, match, mismatch, gap, 'local')
        
        return self.__grid
    
    
    def perform_global_trackback(self):
        """ Tracks traceback cell-wise alignments from the DP grid in accordance
            to the Needleman-Wunsch algorithm, to obtain the optimal global
            aligned header sequences corresponding to Header M & N respectively.
        
        Returns:
            m_alignment (list(str))
            n_alignment (list(str))
        """
        # Since traceback is for global alignment, it starts from last cell...
        last_i = len(self.m_headers)
        last_j = len(self.n_headers)
        last_cell = self.get_cell(last_i, last_j)
        
        # ... and stops at the first cell
        first_cell = self.get_cell(0, 0)
        
        # Keep searching for previous paths until the first cell is reached
        m_alignment = []
        n_alignment = []
        while last_cell != first_cell:
            
            # Store the current alignment steps
            m_alignment.insert(0, last_cell.e1)
            n_alignment.insert(0, last_cell.e2)
            
            # Traceback to previous step
            last_cell = last_cell.prev
            
        return m_alignment, n_alignment
    

    def perform_local_trackback(self):
        """ Tracks traceback cell-wise alignments from the DP grid in accordance
            to the Smith-Waterman algorithm, to obtain all optimal local aligned
            header pairs corresponding to Header M & N respectively.
        
        Returns:
            alignments (list(tuple(list(str))))
        """
        # Since traceback is for local alignment, it starts from the cell
        # with the highest cached score. Therefore, track cells with the 
        # highest scores. There can be multiple possibilities.
        highest_score = max([max([cell.score for cell in row]) 
                             for row in self.__grid])
        
        candidates = []
        for i, row in enumerate(self.__grid):
            for j, col in enumerate(row):
                
                curr_cell = self.get_cell(i, j)
                
                if curr_cell.score == highest_score and curr_cell.score != 0:
                    candidates.append(curr_cell)
                    
        # Perform traceback for each candidate
        alignments = []
        for cell in candidates:
            
            # Keep searching for previous paths until a cell with score 0 is reached
            m_alignment = []
            n_alignment = []
            while cell.score != 0:

                # Store the current alignment steps
                m_alignment.insert(0, cell.e1)
                n_alignment.insert(0, cell.e2)

                # Traceback to previous step
                cell = cell.prev
                
            alignments.append((m_alignment, n_alignment))
            
        return alignments
    
    ##################
    # Core Functions #
    ##################
    
    def align(self, match=10, mismatch=-10, gap=0, algo='global'):
        """ Automates the pairwise alignment process to obtained an optimal
            alignment of specified headers, given specified match-mismatch
            & indel penalties, as well as an appropriate pairwise alignment
            algorithm.

        Args:
            match (float): Reward earned for a correct elementwise match
            mismatch (float): Penalty incurred for a wrong elementwise match
            gap (float): Penalty incurred to open a gap
            algo (str): Algorithm to use for pairwise alignment 
                        Supported algorithms: 1) Global (Needleman-wunsch)
                                              2) Local  (Smith-waterman)
        Returns:
            alignment  (tuple(list(str)))       if algo == global
            alignments (list(tuple(list(str)))) if algo == local
        """
        if algo == 'global':
            self.perform_global_alignment(match, mismatch, gap)
            return self.perform_global_trackback()
        
        elif algo == 'local':
            self.perform_local_alignment(match, mismatch, gap)
            return self.perform_local_trackback()
        
        else:
            raise NotImplementedError("Algorithm not supported!")        
    

#################################################
# Data Alignment Class - MultipleFeatureAligner #
#################################################

class MultipleFeatureAligner:
    """
    Uses a STAR multiple sequence alignment algorithm to align
    under-represented features across datasets. Pairwise alignment
    is first made between all header sequences, after which each
    pair is scored based on their sum of pairs. These scores are
    then used to determine the optimal seeding header sequence
    that is most similar to all other sequences. Lastly, the 
    pairwise alignments that involve the seeding header sequence
    are merged together using a "once a gap, always a gap" rule.

    Keyword Definitions:
        header     - A list of string features
        headers    - A list of header 
        alignment  - A tuple of aligned headers
        alignments - A list of alignment
            
    Attributes:
        headers (list(list(str))): Specified headers to be aligned
        superset (list(str)): Set of all features acoss all headers
        
    Args:
        headers (list(list(str))): Specified headers to be aligned
    """
    
    def __init__(self, headers):
        
        self.headers = headers
        self.superset = sorted(
            list(set().union(*[set(h) for h in headers]))
        )

    ############
    # Checkers #
    ############
    
    @staticmethod
    def is_equivalent(header_1, header_2):
        """ Checks if 2 header sequences are equivalent. Equivalence
            is defined as having the same element frequencies after
            all gaps have been removed
         
        Args:
            header_1 (list(str)): 1st header sequence
            header_2 (list(str)): 2nd header sequence 
        Returns:
            True     if specified headers are equivalent
            False    Otherwise
        """
        filter_out_gaps = lambda x: [element for element in x if element]
        
        composition_1 = Counter(filter_out_gaps(header_1))
        composition_2 = Counter(filter_out_gaps(header_2))
        
        return composition_1 == composition_2
        
        
    def is_part_of(self, header, alignment):
        """ Checks if a header sequence is one of the 2 subjects
            in an alignment
            
        Args:
            header (list(str)): Features headers to be aligned
            alignment (tuple(list(str))): 
        Returns:
            True     if specified header was part of alignment
            False    Otherwise
        """
        equivalences = [self.is_equivalent(header, h) for h in alignment]
        
        return True in equivalences
        
    
    ###########
    # Helpers #
    ###########
        
    @staticmethod
    def calculate_sum_of_pairs(match, mismatch, gap, alignment):
        """ Given a set of match-mismatch & indel penalties, calculate
            MFA evaluation metric, using the Sum-of-Pairs scheme for
            a specified alignment
            
        Args:
            match (float): Reward earned for a correct elementwise match
            mismatch (float): Penalty incurred for a wrong elementwise match
            gap (float): Penalty incurred to open a gap
            alignment (tuple(list(str))): Aligned feature sequences
        Returns:
            Sum-of-Pairs score (float)
        """        
        score = 0
        for column in zip(*alignment):
            
            for pair in itertools.combinations(column, 2):
            
                # Both elements match (i.e. aligned correctly)
                if len(set(pair)) == 1:
                    score += match
                
                # Gaps exists
                elif None in pair:
                    score += gap
                    
                # None of the elements are gaps, but don't align ==> mismatch
                else:
                    score += mismatch
                    
        return score
                    
        
    @staticmethod
    def calculate_concensus_score(alignment):
        """ Calculates MFA evaluation metric via concensus count for a
            specified alignment
        
        Args:
            alignment (tuple(list(str))): Aligned feature sequences
        Returns:
            Concensus score (float)
        """
        score = 0
        for column in zip(*alignment):

            score += Counter(column).most_common(1)[0][1]
            
        return score
    
    
    @staticmethod
    def sort_alignment(template, alignment):
        """ Sorts all columns in a specified alignment according to
            an arbitary order specified in a template.
            
        Args:
            template (list(str)): Template header to be sorted
            alignment (tuple(list(str))): Aligned feature sequences
        Returns:
            Sorted alignment (tuple(list(str)))
        """
        return list(map(list, zip(*sorted(zip(template, *alignment)))))
        
    
    def perform_handshake_alignments(self, match, mismatch, gap, algo):
        """ Pairwise aligns all specified headers with each other in every
            possible combination pair.
            
        Args:
            match (float): Reward earned for a correct elementwise match
            mismatch (float): Penalty incurred for a wrong elementwise match
            gap (float): Penalty incurred to open a gap
            algo (str): Algorithm to use for pairwise alignment 
                        Supported algorithms: 1) Global (Needleman-wunsch)
                                              2) Local  (Smith-waterman)
        Returns:
            Handshake Alignments (list(tuple(list(str))))
        """
        handshake_alignments = []
        for combination in itertools.combinations(self.headers, 2):

            ###########################
            # Implementation Footnote #
            ###########################

            # [Cause]
            # In pairwise alignment, order of M & N sequences specified will
            # affect the order where spacers are to be inserted. 
            # eg. ['cp_1', 'cp_2', 'cp_3', 'cp_4', 'exang_0', 'exang_1', 'fbs_0', 'fbs_1',  None,      None,      None,    None  ]
            #     [ None,  'cp_2', 'cp_3', 'cp_4',  None,      None,      None,    None,   'exang_3', 'exang_4', 'fbs_3', 'fbs_4']
            #     vs
            #     ['cp_1', 'cp_2', 'cp_3', 'cp_4',  None,      None,      None,    None,   'exang_0', 'exang_1', 'fbs_0', 'fbs_1']
            #     [ None,  'cp_2', 'cp_3', 'cp_4', 'exang_3', 'exang_4', 'fbs_3', 'fbs_4',  None,      None,      None,    None  ]
            
            # [Problems]
            # This results compounded insertions when combining pairwise 
            # alignments in MFA, since each variant is considered to be unique
            # resulting in redundant insertions
            # eg. Seeding pairs:
            #  1) ['cp_1', 'cp_2', 'cp_3', 'cp_4', 'exang_0', 'exang_1', 'fbs_0', 'fbs_1',  None,      None,      None,    None  ]
            #     [ None,  'cp_2', 'cp_3', 'cp_4',  None,      None,      None,    None,   'exang_3', 'exang_4', 'fbs_3', 'fbs_4']
            #  2) ['cp_1', 'cp_2', 'cp_3', 'cp_4', 'exang_0', 'exang_1', 'fbs_0', 'fbs_1',  None,      None,      None,    None  ]
            #     ['cp_1', 'cp_2', 'cp_3', 'cp_4',  None,      None,      None,    None,   'exang_0', 'exang_1', 'fbs_0', 'fbs_1']
            #  3) ['cp_1', 'cp_2', 'cp_3', 'cp_4',  None,      None,      None,    None,   'exang_0', 'exang_1', 'fbs_0', 'fbs_1']
            #     [ None,  'cp_2', 'cp_3', 'cp_4', 'exang_3', 'exang_4', 'fbs_3', 'fbs_4',  None,      None,      None,    None  ]
            #     Resultant MFA:
            #     ['cp_1', 'cp_2', 'cp_3', 'cp_4', None, None, None, None, 'exang_0', 'exang_1', 'fbs_0', 'fbs_1', None,      None,      None,    None  ]
            #     [ None,  'cp_2', 'cp_3', 'cp_4', None, None, None, None,  None,      None,      None,    None,  'exang_3', 'exang_4', 'fbs_3', 'fbs_4']
            #     ['cp_1', 'cp_2', 'cp_3', 'cp_4', None, None, None, None, 'exang_0', 'exang_1', 'fbs_0', 'fbs_1', None,      None,      None,    None  ]

            # [Solution]
            # Explicitly ensure that all pairwise combinations are first sorted!

            # Sort combination before performing pairwise alignment
            sorted_combination = sorted(combination)
            pwf_aligner = PairwiseFeatureAligner(*sorted_combination)
            
            pw_alignment = pwf_aligner.align(
                match=match, 
                mismatch=mismatch, 
                gap=gap,
                algo=algo
            )
            
            if len(pw_alignment) > 0:
                
                if algo == "global":
                    handshake_alignments.append(pw_alignment)
                    
                elif algo == 'local':
                    handshake_alignments += pw_alignment
        
        return handshake_alignments
    
    
    def find_seeding_header(self, alignments, scores):
        """ Given a set of pairwise alignments & their respective
            MFA benchmarking scores, calculate the distance between
            each pair and every other pair, and return the header
            sequence that is most similar to every other header.
        
        Args:
            alignments (list(tuple(list(str)))): Aligned feature sequences
            scores (float): MFA metric calculated for each header sequence
        Returns:
            Seeding header sequence (list(str))
        """
        # Total up all scores for alignments involving each header
        scored_headers = {}
        for header in self.headers:
            
            header_score = 0
            for alignment, score in zip(alignments, scores):
                
                # Update score if current alignment is relevant
                if self.is_part_of(header, alignment):
                    header_score += score
                    
            scored_headers[tuple(header)] = header_score
            
        # Find the header with the highest score
        seeding_header = max(scored_headers, key=scored_headers.get)
                
        return list(seeding_header)
                    

    def extract_relevant_alignments(self, header, alignments):
        """ Extract all alignments involving the seeding header.
            Subject header is always stored at 1st index
        
        Args:
            header (list(str)): Features headers to be aligned
            alignments (list(tuple(list(str)))): Aligned feature sequences
        Returns
            Sorted alignments (list(tuple(list(str))))
        """
        
        return  [
            sorted(
                alignment, 
                key=lambda x: self.is_equivalent(x, header),
                reverse=True
            ) for alignment in alignments
            if self.is_part_of(header, alignment)
        ]
            
        
    def merge_alignments(self, alignments, scores):
        """ Given a set of handshake alignments & their respective MFA
            scores, find the seeding header & merge all relevant pairs
            into a single multiple feature alignment
        
        Args:
            alignments (list(tuple(list(str)))): Aligned feature sequences
            scores (float): MFA metric calculated for each header sequence
        Returns:
            Multiple feature alignment (tuple(list(str)))
        """
        seeding_header = self.find_seeding_header(alignments, scores)
        
        relevant_alignments = self.extract_relevant_alignments(
            seeding_header,
            alignments
        )
        
        final_alignments = []
        for alignment in relevant_alignments:
            
            root_header = alignment[0]
            side_header = alignment[1]
        
            if len(root_header) < len(seeding_header):
                template_header = root_header
                
            else:
                template_header = seeding_header
                
            limit = max(len(root_header), len(seeding_header))
                
            j = 0
            while j < limit:
                
                try:
                    curr_root_element = root_header[j]
                    curr_side_element = side_header[j]

                    # Element-gap
                    if curr_root_element and not seeding_header[j]:
                        root_header.insert(j, None)
                        side_header.insert(j, None)

                    # Gap-element
                    elif not curr_root_element and seeding_header[j]:
                        seeding_header.insert(j, None)
                        [a.insert(j, None) for a in final_alignments]
                
                except IndexError:
                    # IndexError occurs because
                    template_header.append(None)
                    
                    # Current alignment is missing tail-end gaps, so
                    if template_header is root_header:
                        side_header.append(None)
                    
                    else:
                        [a.append(None) for a in final_alignments]
                
                j += 1
            
            final_alignments.append(side_header)
        
        return tuple([seeding_header] + final_alignments)
    
    
    def arrange_alignment(self, mf_alignment):
        """ Rearrange augmented header sequences in a multiple feature 
            alignment to match the orignal input order.
            
        Args:
            mf_alignment (tuple(list(str))): Multiple feature alignment
        Returns:
            Ordered MFA (tuple(list(str)))
        """
        ordered_mf_alignment = []
        for raw_header in self.headers:
            
            for augmented_header in mf_alignment:
                
                if self.is_equivalent(raw_header, augmented_header):
                    
                    ordered_mf_alignment.append(augmented_header)
                    break
                    
        return ordered_mf_alignment
                    
    ##################
    # Core Functions #
    ##################
    
    def align(self, match=10, mismatch=-10, gap=0, algo='global'):
        """ Automates the STAR MFA process to obtained a multiple feature
            alignment of specified headers, given specified match-mismatch
            & indel penalties, as well as an appropriate pairwise alignment
            algorithm.
        
        Args:
            match (float): Reward earned for a correct elementwise match
            mismatch (float): Penalty incurred for a wrong elementwise match
            gap (float): Penalty incurred to open a gap
            algo (str): Algorithm to use for pairwise alignment 
                        Supported algorithms: 1) Global (Needleman-wunsch)
                                              2) Local  (Smith-waterman)
        Returns:
            Multiple feature alignment (tuple(list(str)))
        """    
        handshake_alignments = self.perform_handshake_alignments(
            match=match,
            mismatch=mismatch,
            gap=gap,
            algo=algo
        )

        alignment_scores = [
            self.calculate_sum_of_pairs(
                match=match,
                mismatch=mismatch,
                gap=gap,
                alignment=alignment
            )
            for alignment in handshake_alignments
        ]
        
        mf_alignment = self.merge_alignments(
            handshake_alignments, 
            alignment_scores
        )

        ordered_mf_alignment = self.arrange_alignment(mf_alignment)
        return ordered_mf_alignment
        
#########        
# Tests #
#########

if __name__ == "__main__":
    X_data_headers = [
        ['age', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'chol', 'cp_1', 'cp_2', 'cp_3', 'cp_4', 'exang_0', 'exang_1', 'fbs_0', 'fbs_1', 'oldpeak', 'restecg_0', 'restecg_1', 'restecg_2', 'sex_0', 'sex_1', 'slope_1', 'slope_2', 'slope_3', 'thal_3', 'thal_6', 'thal_7', 'thalach', 'trestbps'],
        ['age', 'ca_0', 'ca_1', 'chol', 'cp_2', 'cp_3', 'cp_4', 'exang_0.0', 'exang_1.0', 'fbs_0.0', 'fbs_1.0', 'oldpeak', 'restecg_0', 'restecg_1', 'restecg_2', 'sex_0', 'sex_1', 'slope_1', 'slope_2', 'slope_3', 'thal_3', 'thal_6', 'thal_7', 'thalach', 'trestbps'], 
        ['age', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'chol', 'cp_1', 'cp_2', 'cp_3', 'cp_4', 'exang_0', 'exang_1', 'fbs_0', 'fbs_1', 'oldpeak', 'restecg_0', 'restecg_1', 'restecg_2', 'sex_0', 'sex_1', 'slope_1', 'slope_2', 'slope_3', 'thal_3', 'thal_6', 'thal_7', 'thalach', 'trestbps']
    ]

    X_mfa_aligner = MultipleFeatureAligner(headers=X_data_headers)
    X_mf_alignments = X_mfa_aligner.align()

    print(X_mf_alignments)

    """
    [
        ['age', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'chol', 'cp_1', 'cp_2', 'cp_3', 'cp_4', None, None, None, None, 'exang_0', 'exang_1', 'fbs_0', 'fbs_1',  None,        None,        None,      None,     'oldpeak', 'restecg_0', 'restecg_1', 'restecg_2', 'sex_0', 'sex_1', 'slope_1', 'slope_2', 'slope_3', 'thal_3', 'thal_6', 'thal_7', 'thalach', 'trestbps'], 
        ['age', 'ca_0', 'ca_1',  None,   None,  'chol',  None,  'cp_2', 'cp_3', 'cp_4', None, None, None, None,  None,      None,      None,    None,   'exang_0.0', 'exang_1.0', 'fbs_0.0', 'fbs_1.0', 'oldpeak', 'restecg_0', 'restecg_1', 'restecg_2', 'sex_0', 'sex_1', 'slope_1', 'slope_2', 'slope_3', 'thal_3', 'thal_6', 'thal_7', 'thalach', 'trestbps'], 
        ['age', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'chol', 'cp_1', 'cp_2', 'cp_3', 'cp_4', None, None, None, None, 'exang_0', 'exang_1', 'fbs_0', 'fbs_1',  None,        None,        None,      None,     'oldpeak', 'restecg_0', 'restecg_1', 'restecg_2', 'sex_0', 'sex_1', 'slope_1', 'slope_2', 'slope_3', 'thal_3', 'thal_6', 'thal_7', 'thalach', 'trestbps']
    ]

    """
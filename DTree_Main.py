"""
Machine Learning Assignment 1 Solution Code:
    Creation of Decision Tree using  variance and information gain heuristic. Testing and comparing
    the accuracies using post-pruning methods given in the problem statement.
Author: Manik Narang [Graduate Student MS CS (Fall 2018)]
Net ID: mxn170011
Email : mxn170011@utdallas.edu
UTD ID: 2021424049
Professor: Anjum Chida
Requires: python 3.7.0
External Library: pandas
Internal Libraries: sys,math,copy,random,time
Running Instructions:
    Command : python3 DTree_Main.py <lvalue> <kvalue> <training data csv file path> <validation data csv file path> <test data csv file path> <toprint:yes/no>
"""
import sys
import pandas
import math
import copy
import random
import time
"""
Globals:
    indent : for printing decision tree in the correct format, indent counter adds and subtracts,providing proper indentation
    node_counter_entropy : used for node numbering when entropy heuristic is used for decision tree creation
    node_counter_variance : used for node numbering when variance heuristic is used for decision tree creation
"""
indent = 0
node_counter_entropy = 0
node_counter_variance = 0


class DTreeNode(object):
    """
    Class : DTreeNode
        Primary Decision Tree Class, Instance of this class will hold the root and other nodes of the tree
    Properties:
        left: Contain the DTreeNode object for left side, when decision made is 0
        right: Contain the DTreeNode object for the right side, when decision made is 1
        data: Contain the root value i.e. Attribute Value or leaf node value[1 or 0]
    """
    def __init__(self, left=None, right=None, node_number=None):
        self.left = left
        self.right = right
        self.data = None
        self.node_number = node_number


class Attribute(object):
    """
    Class: Attribute
        Attribute Class containing positive and negative example number count,names and other important heuristic functions
    Properties:
        positives: hold the positive example count for the attribute or attribute value such as XB or XB1,XB0
        negatives: hold the negative example count for the attribute or attribute value such as XB or XB0,XB1
        attributeName: hold the attribute name eg. XB, XC etc.
    Methods:
        get_entropy():
            Returns entropy for the given attribute using positive and negative example counts
            :return:
                entropy : float
                    entropy of the given attribute
        get_total():
            Returns the sum of positive and negative examples
            :return:
                total : int
                    sum of positive and negative examples
        get_variance():
            Returns variance for the given attribute using positive and negative example counts
            :return:
                variance: float
                    variance of the given attribute
    """
    def __init__(self):
        self.positives = 0
        self.negatives = 0
        self.attributeName = str()

    def get_entropy(self):
        if self.positives == 0 or self.negatives == 0:
            return 0
        else:
            return entropy_calc(self.positives, self.negatives)

    def get_total(self):
        return float(self.positives + self.negatives)

    def get_variance(self):
        if self.positives == 0 or self.negatives == 0:
            return 0
        else:
            return variance_calc(self.positives,self.negatives)


def entropy_calc(positives, negatives):
    """
        Calculates Entropy of the given negative and positive example set
        :param positives: int : Number of positive examples in the attribute
        :param negatives: int : Number of negative examples in the attribute
        :return: entropy : float : Entropy of the given positive and negative example set
    """
    total = float(positives + negatives)
    positive_fraction = float(positives/total)
    negative_fraction = float(negatives/total)
    entropy = (-1 * positive_fraction * math.log(positive_fraction, 2)) - (negative_fraction * math.log(negative_fraction, 2))
    return float(entropy)


def variance_calc(positives, negatives):
    """
        Calculates Variance of the given negative and positive example set
        :param positives: int : Number of positive examples in the attribute
        :param negatives: int : Number of negative examples in the attribute
        :return: variance: float : Variance of the given positive and negative example set
    """
    total = float(positives + negatives)
    positive_fraction = float(positives/total)
    negative_fraction = float(negatives/total)
    variance = positive_fraction * negative_fraction
    return float(variance)


def variance_gain_calc(primary_attribute, secondary_attribute_list):
    """
        Calculates the net variance information gain with respect to some primary attribute Variance_Information_Gain(S,A)
        :param primary_attribute: Attribute(): The attribute object with whom the relative variance information gain would be calculated
        :param secondary_attribute_list: list(Attribute) : The attribute value list (eg.XB0 or XB1) of we have to calculate the gain
        :return: relative_gain: float : The relative gain of the attribute with respect to primary
    """
    net_secondary_variance = float()
    for each_attribute_item in secondary_attribute_list:
        fraction_each_attribute = each_attribute_item.get_total() / primary_attribute.get_total()
        net_secondary_variance += fraction_each_attribute * each_attribute_item.get_variance()
    gain = primary_attribute.get_variance() - net_secondary_variance
    return float(gain)


def gain_calc(primary_attribute, secondary_attribute_list):
    """
        Calculates the net entropy information gain with respect to some primary attribute Entropy_Information_Gain(S,A)
        :param primary_attribute: Attribute(): The attribute object with whom the relative entropy information gain would be calculated
        :param secondary_attribute_list: list(Attribute) : The attribute value list (eg.XB0 or XB1) of we have to calculate the gain
        :return: relative_gain: float : The relative gain of the attribute with respect to primary
    """
    net_secondary_entropy = float()
    for each_attribute_item in secondary_attribute_list:
        fraction_each_attribute = each_attribute_item.get_total() / primary_attribute.get_total()
        net_secondary_entropy += fraction_each_attribute * each_attribute_item.get_entropy()
    gain = primary_attribute.get_entropy() - net_secondary_entropy
    return float(gain)


def ID3(training_data_dataframe, target_attribute, primary_attribute, attribute_name_list, heuristic):
    """
        Uses ID3 Algorithm to create decision tree based on given information gain heuristic
        :param training_data_dataframe: pandas.Dataframe() :training data frame set with attribute values and class results
        :param target_attribute: Attribute() : target attribute for which we have to create the decision tree
        :param primary_attribute: Attribute(): primary attribute with respect to which new sub-tree would be created recursively
        :param attribute_name_list: list(str): list with respect to which new sub-tree would be chosen upon
        :param heuristic: str : 'Entropy' or 'Variance' heuristic from which the attributes would be chosen
        :return: root : DTreeNode() :recursive root of the decision tree or sub-tree
    """
    global node_counter_entropy
    global node_counter_variance
    root = DTreeNode()
    # For all positive examples return 1
    if calc_target_attribute_value(training_data_dataframe, target_attribute, 0) == 0:
        root.data = 1
        return root
    # For all negative examples return 0
    if calc_target_attribute_value(training_data_dataframe, target_attribute, 1) == 0:
        root.data = 0
        return root
    # If Attribute list is fully exhausted return the majority class[target attribute] as leaf node
    if len(attribute_name_list) == 0:
        root.data = 1 if calc_target_attribute_value(training_data_dataframe, target_attribute, 1) > calc_target_attribute_value(training_data_dataframe, target_attribute, 0) else 0
        return root
    # Get the best attribute based on the given heuristic that classify the examples as spread out or far as possible
    selected_best_attribute_name = get_best_attribute(training_data_dataframe, primary_attribute, attribute_name_list, heuristic)

    # Number the non-leaf nodes as they are created based on each heuristic
    root.data = selected_best_attribute_name
    if str.upper(heuristic) == 'ENTROPY':
        node_counter_entropy += 1
        root.node_number = node_counter_entropy
    else:
        node_counter_variance += 1
        root.node_number = node_counter_variance

    # Remove the best selected attribute from the list for creating an attribute list for the next recursive sub-tree
    attribute_name_list.remove(selected_best_attribute_name)
    new_attribute_list1 = copy.deepcopy(attribute_name_list)
    new_attribute_list2 = copy.deepcopy(attribute_name_list)

    # Populate the 1 value Attribute of the best selected attribute
    selected_best_attribute_value_1 = calc_attribute_value_counter(training_data_dataframe, selected_best_attribute_name, target_attribute, 1)
    # Get training data frame subset for the all the 1s of the best selected attribute
    training_data_dataframe_subset_1 = filter_training_data(training_data_dataframe, selected_best_attribute_name, 1)
    # Recursively create right node for all the 1s of the best selected attribute
    root.right = ID3(training_data_dataframe_subset_1, target_attribute, selected_best_attribute_value_1, new_attribute_list1, heuristic)
    # Populate the 0 value Attribute of the best selected attribute
    selected_best_attribute_value_0 = calc_attribute_value_counter(training_data_dataframe, selected_best_attribute_name, target_attribute, 0)
    # Get training data frame subset for the all the 0s of the best selected attribute
    training_data_dataframe_subset_0 = filter_training_data(training_data_dataframe, selected_best_attribute_name, 0)
    # Recursively create right node for all the 0s of the best selected attribute
    root.left = ID3(training_data_dataframe_subset_0, target_attribute,selected_best_attribute_value_0, new_attribute_list2, heuristic)
    return root


def filter_training_data(training_data_dataframe, selected_attribute_name, target_value):
    """
    filters the training data according to the attribute and attribute value passed,the method creates a subset
    from the training dataframe,eg,filter all training data where XB=1 etc.
    :param training_data_dataframe: pandas.Dataframe():the training data set
    :param selected_attribute_name: str: the attribute upon which the data needs to be filtered
    :param target_value: value of attribute for which the data needs to be filtered
    :return: training_data_subset: pandas.dataframe(): training data subset according to the attribute given
    """
    queryString = str(str.strip(selected_attribute_name) + '== ' + str(target_value))
    training_data_subset = training_data_dataframe.query(queryString)
    return training_data_subset


def calc_target_attribute_value(training_data_dataframe, target_attribute, target_value):
    """
    counts the attribute value number for the given target attribute and target value
    :param training_data_dataframe: pandas.Dataframe(): the dataframe from which the target value is to be counted
    :param target_attribute: Attribute() : target attribute object whose value needs to be counted in the training data set
    :param target_value: int : the attribute value to be counted,0 or 1
    :return: int: count of the attribute value
    """
    count = int()
    for index,row in training_data_dataframe.iterrows():
        if int(row[target_attribute.attributeName]) == target_value:
            count += 1
    return count


def calc_attribute_value_counter(training_data_dataframe, attribute_name ,target_attribute, target_value):
    """
    Calculates and populates the attribute with the number of positive examples or negative examples it has for a
    particular target attribute
    :param training_data_dataframe : pandas.Dataframe():the training data set from which the values will be counted
    :param attribute_name: str : the name of the attribute of which we need to calculate positives and negatives
    :param target_attribute: Attribute():target attribute object with respect to which we have to calculate value
    :param target_value: value of the attribute of which we need to calculate positives and negatives
    :return:resultant_attribute_object:Attribute(): populated attribute object
    """
    positives = int()
    negatives = int()
    for index,row in training_data_dataframe.iterrows():
        if int(row[attribute_name]) == target_value and int(row[target_attribute.attributeName]) == 1:
            positives += 1
        if int(row[attribute_name]) == target_value and int(row[target_attribute.attributeName]) == 0:
            negatives += 1
    new_attribute = Attribute()
    new_attribute.positives = positives
    new_attribute.negatives = negatives
    return new_attribute


def get_best_attribute(training_data_dataframe, primary_attribute, attribute_name_list, heuristic):
    """
    Returns the best attribute based on the training data  with respect to the primary attribute and the heuristic string passed
    :param training_data_dataframe: pandas.dataframe() : the training data set from which the best attribute will be returned
    :param primary_attribute: Attribute(): the attribute with respect to which the best attribute's heuristic will be calculated
    :param attribute_name_list:list(str) the list of attributes that would be compared with each other and one would be selected
    :param heuristic: str : Entropy or Variance heuristic
    :return: attribute_name:str: best selected attribute based on the given heuristic
    """
    if str.upper(heuristic) == 'ENTROPY':
        return get_best_attribute_entropy(training_data_dataframe, primary_attribute, attribute_name_list)
    else:
        return get_best_attribute_variance(training_data_dataframe, primary_attribute, attribute_name_list)


def get_best_attribute_entropy(training_data_dataframe, primary_attribute, attribute_name_list):
    """
    Calculates the best attribute based on entropy heuristic with respect to the primary attribute
    :param training_data_dataframe: pandas.dataframe():the training data set from which the best attribute will be calculated
    :param primary_attribute: Attribute(): the attribute with respect to which the best attribute's heuristic will be calculated
    :param attribute_name_list:list(str): the attribute list from which the best attribute will be selected
    :return:best_attribute_name_entropy: str: best attribute name based on the entropy heuristic
    """
    gain = -999999
    selected_best_attribute = str()
    for each_attribute in attribute_name_list:
        attribute_value_0 = get_attribute_value(training_data_dataframe, each_attribute, 0)
        attribute_value_1 = get_attribute_value(training_data_dataframe, each_attribute, 1)
        secondary_attribute_value_list = [attribute_value_0,attribute_value_1]
        if float(gain_calc(primary_attribute,secondary_attribute_value_list)) > float(gain):
            gain = gain_calc(primary_attribute,secondary_attribute_value_list)
            selected_best_attribute = copy.deepcopy(each_attribute)
    return selected_best_attribute


def get_attribute_value(training_data_dataframe, each_attribute, possible_value):
    """
    Based on training data populates the attribute object with positive and negative examples
    :param training_data_dataframe: pandas.dataframe():training data set that helps to populate the attribute
    :param each_attribute:str: the attribute name that needs to be populated
    :param possible_value:int: the attribute value whos negative and positive examples we need to count
    :return: populated_attribute_object: Attribute(): resultant populated attribute object
    """
    positives = int()
    negatives = int()
    for index, row in training_data_dataframe.iterrows():
        if row[each_attribute] == possible_value and row['Class'] == 1:
            positives += 1
        if row[each_attribute] == possible_value and row['Class'] == 0:
            negatives += 1
    attribute_value = Attribute()
    attribute_value.positives = positives
    attribute_value.negatives = negatives
    return attribute_value


def get_best_attribute_variance(training_data_dataframe, primary_attribute, attribute_name_list):
    """
    Calculates the best attribute based on variance heuristic with respect to the primary attribute
        :param training_data_dataframe: pandas.dataframe():the training data set from which the best attribute will be calculated
        :param primary_attribute: Attribute(): the attribute with respect to which the best attribute's heuristic will be calculated
        :param attribute_name_list:list(str): the attribute list from which the best attribute will be selected
        :return:best_attribute_name_variance: str: best attribute name based on the variance heuristic
    """
    gain = -999999999
    selected_best_attribute = str()
    for each_attribute in attribute_name_list:
        attribute_value_0 = get_attribute_value(training_data_dataframe, each_attribute, 0)
        attribute_value_1 = get_attribute_value(training_data_dataframe, each_attribute, 1)
        secondary_attribute_value_list = [attribute_value_0,attribute_value_1]
        if float(variance_gain_calc(primary_attribute,secondary_attribute_value_list)) > float(gain):
            gain = variance_gain_calc(primary_attribute,secondary_attribute_value_list)
            selected_best_attribute = copy.deepcopy(each_attribute)
    return selected_best_attribute


def get_prediction_accuracy(testing_data_dataframe, tree_root_node):
    """
    Calculates the prediction accuracy percentage based on the test data given
    :param testing_data_dataframe: pandas.dataframe(): test data to calculate the prediction accuracy
    :param tree_root_node: DTreeNode(): root node of tree for which we have to calculate the prediction accuracy
    :return: accuracy_percentage, error_percentage: float: percentages for the corresponding tree
    """
    correct_answer_count = int()
    wrong_answer_count = int()
    total_answer_count = int()
    tree_root_node_copy = copy.deepcopy(tree_root_node)
    for index, row in testing_data_dataframe.iterrows():
        tree_root_node = copy.deepcopy(tree_root_node_copy)
        while tree_root_node.data != 0 and tree_root_node.data != 1:

            if row[tree_root_node.data] == 0:
                if tree_root_node.left is not None:
                    tree_root_node = tree_root_node.left
                else:
                    break
            else:
                if tree_root_node.right is not None:
                    tree_root_node = tree_root_node.right
                else:
                    break

        if tree_root_node.data == row['Class']:
            correct_answer_count += 1
        else:
            wrong_answer_count += 1
        total_answer_count += 1
    accuracy_percentage = (correct_answer_count/total_answer_count) * 100
    error_percentage = (wrong_answer_count/total_answer_count) * 100
    return accuracy_percentage, error_percentage


def post_pruning(DTree, l_number, k_number, heuristic,validation_data_dataframe):
    """
    Based on the pruning algorithm finds the best decision tree using validation dataset
    :param DTree: DTreeNode(): The tree to prune
    :param l_number:int: L Integer for Algorithm
    :param k_number: int: K Integer for Algorithm
    :param heuristic: Entropy or Variance Heuristic
    :param validation_data_dataframe: pandas.dataframe() :Validation Data frame for the validating and using in pruning algorithm
    :return: Best Decision Tree: DTreeNode(), Changed_flag: bool: To know whether the pruning algorithm helped or not
    """
    DTree_best = copy.deepcopy(DTree)
    changed = False
    i = 1
    j = 1
    accuracy_DTree_best, error_DTree_best = get_prediction_accuracy(validation_data_dataframe, DTree_best)
    for i in range(l_number):
        DTree_dash = copy.deepcopy(DTree)
        m = random.randint(1, k_number)
        for j in range(m):
            if str.upper(heuristic) == 'ENTROPY':
                n = node_counter_entropy
            else:
                n = node_counter_variance
            p = random.randint(1, n)
            DTree_dash = prune_tree(DTree_dash, p)
        accuracy_DTree_dash, error_DTree_dash = get_prediction_accuracy(validation_data_dataframe, DTree_dash)
        if accuracy_DTree_dash > accuracy_DTree_best:
            DTree_best = copy.deepcopy(DTree_dash)
            changed = True
    return DTree_best, changed


def prune_tree(DTree, p):
    """
    Tree pruning based on pruning algorithm, finding subtree based on p number and pruning it and returning back the tree
    :param DTree: DTreeNode():The tree to be pruned
    :param p: int: The pth Node of the tree that needs to be pruned to a leaf node
    :return: new pruned tree :DTreeNode()
    """
    subtree = traverse(DTree, p)
    count_0 = calc_majority_class(subtree, 0)
    count_1 = calc_majority_class(subtree, 1)
    value_to_replace = 1 if count_1 > count_0 else 0
    return traverse_and_prune(DTree, p, value_to_replace)


def calc_majority_class(subtree, value):
    """
    Checks the Majority class in the sub tree
    :param subtree: DTreeNode(): subtree whos majority class needs to be checked
    :param value: int : value of the leaf node who's majority needs to be checked
    :return:count: int : count of the value of the given leaf node
    """
    if subtree is None:
        return 0
    if subtree.left is None and subtree.right is None and subtree.data == value:
        return 1
    else:
        return calc_majority_class(subtree.left, value) + calc_majority_class(subtree.right, value)


def traverse(DTree, p):
    """
    Traverses the tree and finds the pth subtree root
    :param DTree: DTreeNode():The decision tree to be traversed
    :param p: int : The node number for which the subtree needs to be returned
    :return: subtree:DTreeNode(): The pth node subtree
    """
    if DTree is not None:
        if DTree.node_number == p:
            return DTree
        else:
            foundTree = traverse(DTree.left, p)
            if foundTree is None:
               foundTree = traverse(DTree.right, p)
            return foundTree
    else:
        return None


def traverse_and_prune(Dtree, p, value_to_replace):
    """
    Traverses and prunes the pth subtree and replaces it with the majority class
    :param Dtree: DTreeNode() : The tree to be traversed and pruned
    :param p: The pth Node that need to be searched and pruned
    :param value_to_replace: majority class value that needs to be replaced by the method
    :return: pruned_tree:DTreeNode(): resultant pruned tree
    """
    def traverse_prune(DNode):
        if DNode is not None:
            if DNode.node_number == p:
                DNode.data = value_to_replace
                DNode.left = None
                DNode.right = None
                return
            else:
                foundTree = traverse_prune(DNode.left)
                if foundTree is None:
                    foundTree = traverse_prune(DNode.right)
                return 0
        else:
            return None
    traverse_prune(Dtree)
    return Dtree


def print_tree(rootNode):
    """
    Printing the tree with the specified format
    :param rootNode: DTreeNode():The root node from where the tree originates
    :return: nothing
    """
    global indent
    i = 0
    if rootNode.left is not None:
        for i in range(indent):
            print('| ', end='')
        if rootNode.left.data == 0 or rootNode.left.data == 1:
            print(rootNode.data + ' : 0', end='')
        else:
            print(rootNode.data + ' : 0')
        indent += 1
        print_tree(rootNode.left)
        indent -= 1
    if rootNode.right is not None:
        for i in range(indent):
            print('| ', end='')
        if rootNode.right.data == 0 or rootNode.right.data == 1:
            print(rootNode.data + ' : 1', end='')
        else:
            print(rootNode.data + ' : 1')
        indent += 1
        print_tree(rootNode.right)
        indent -= 1
    if rootNode.right is None or rootNode.left is None:
        print(' : ' + str(rootNode.data))


if __name__ == '__main__':
    # Get arguments from command line
    timestr = time.strftime("%Y%m%d-%H%M%S")
    lValue = 1
    kValue = 2
    training_set_data = pandas.read_csv('data_sets1/training_set.csv')
    validation_set_data = pandas.read_csv('data_sets1/validation_set.csv')
    test_set_data = pandas.read_csv('data_sets1/test_set.csv')
    to_print = True
    # Check print flag to print on console or a file
    if not to_print:
        sys.stdout = open('file_'+timestr+'.txt', 'w+')
    # Initialize Starting Values for Decision Tree : Entropy Heuristic
    test_attribute = Attribute()
    test_attribute.attributeName = 'Class'
    # Initialize Target Attribute Object : Class
    target_attribute = calc_attribute_value_counter(training_set_data, "Class", test_attribute, 1)
    target_attribute.attributeName = 'Class'
    # Initialize Attribute List
    attribute_list = ['Pitch_Quality', 'More_than_2_Corners', 'Star_Player_Injured', 'Coach_had_poor_season', 'Previous_match_lost', 'previous_2_matches_lost', 'won_last_game',
                      'won_2_last_games', 'star_player_scored_3', 'player_got_red_carded', 'player_got_yellow_carded',
                      'temperature_above_80', 'transfers', 'referee_quality_poor', 'against_top_6', 'XQ', 'XR', 'XS', 'XT', 'XU']
    print('Creating Decision Tree based on Information gain(Entropy)....Please wait a moment...')
    # Run ID3 Algorithm and create decision tree for entropy heuristic
    root = ID3(training_set_data, target_attribute, target_attribute, attribute_list, 'Entropy')
    # Print decision tree with entropy heuristic
    print('Decision Tree based on Information Gain(Entropy) heuristic: ')
    print_tree(root)

    print('Calculating prediction accuracies for the trees...Please wait a moment...')
    # Calculate Decision Tree accuracy percentage : Entropy Heuristic
    entropy_tree_accuracy_percentage, entropy_tree_error_percentage = get_prediction_accuracy(test_set_data, root)

    print('Information Gain(Entropy) Decision Tree Accuracy Percentage with test set[Before Pruning]: ' + str(entropy_tree_accuracy_percentage) + '%')
    print('Information Gain(Entropy) Decision Tree Error Percentage with test set[Before Pruning]: ' + str(entropy_tree_error_percentage) + '%')



















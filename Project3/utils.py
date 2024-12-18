import copy
import re

from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix
from matplotlib import pyplot as plt

def parse_organism_metadata_from_sequence(sequence):
    return {
        'id': sequence.id,
        'name': re.sub(r'\[.*\]', '', sequence.description).replace(sequence.id, '').strip(),
        'organism': re.search(r'\[(.*)\]', sequence.description).group(1),
        'sequence': sequence
    }


def create_sub_dm_with_organism_labels(parent_dm, members):
    sub_dm = DistanceMatrix(names=[member['id'] for member in members])
    for member1 in members:
        for member2 in members:
            name1 = member1['id']
            name2 = member2['id']

            sub_dm[name1, name2] = parent_dm[name1, name2]

    # Rename the nodes to organism names
    sub_dm.names = [member['organism'] for member in members]

    return sub_dm


def draw_group_tree(tree):
    plt.figure(figsize=(10, 10))
    Phylo.draw(
        tree,
        axes=plt.gca(),
        branch_labels=lambda c: round(c.branch_length, 3),
        label_func=lambda n: n.name if n.is_terminal() else '',
        do_show=False
    )

    return tree


def draw_full_tree(tree, records_metadata):
    plt.figure(figsize=(30, 30))
    Phylo.draw(
        tree,
        axes=plt.gca(),
        label_func=lambda n: (
            f"{records_metadata[n.name]['name']} {records_metadata[n.name]['organism']}"
            if n.is_terminal()
            else ''
        ),
        do_show=False
    )


def color_organism_labeled_tree(tree, organism_colors):
    tree_cloned = copy.deepcopy(tree)

    for clade in tree_cloned.find_clades():
        if clade.name in organism_colors:
            clade.color = organism_colors[clade.name]

    return tree_cloned

def color_record_id_labeled_tree_by_organism(tree, record_id_by_organizm_colors):
    tree_cloned = copy.deepcopy(tree)

    for clade in tree_cloned.find_clades():
        if clade.name in record_id_by_organizm_colors:
            clade.color = record_id_by_organizm_colors[clade.name]

    return tree_cloned


def color_record_id_labeled_tree_by_group(tree, record_id_by_group_colors):
    tree_cloned = copy.deepcopy(tree)

    for clade in tree_cloned.find_clades():
        if clade.name in record_id_by_group_colors:
            clade.color = record_id_by_group_colors[clade.name]

    return tree_cloned

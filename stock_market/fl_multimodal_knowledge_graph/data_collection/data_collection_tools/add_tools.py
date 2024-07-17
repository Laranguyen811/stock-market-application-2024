import uri_tools
from uri_tools import uri_to_label
import json

def add_entity(info,graph):
    """ Takes an entity and adds it to a graph.
    Inputs:
    info(dict): A dictionary of information about the entity.
    Returns:
    graph(graph): the graph with the added entity.
    """
    label = uri_to_label(info["@id"]) #turning a URI into a label for an entity with @id(unique id for JASON-ID)
    safe_label = label.replace("'","\\'") #a safe label to handle the replacement of single quotes
    code = f"MERGE (n:Entity {{name: '{safe_label}'}})" #Cyper query to merge an entity wit the given label into the graph
    if 'multi-model' in info:
        for key in ['audio','image','visual']:
            if key in info['multi-model']:
               value = json.dumps(info['multi-model'][key]) #converting the key of the value into a JASON formatted string.
               safe_value = value.replace("'","\\'")
               code += f" SET n.{key} = '{safe_value}'"

from stock_market.fl_multimodal_knowledge_graph.data_collection.data_collection_tools.uri_tool import uri_to_label
import json


def add_entity(info,graph):
    """ Takes an entity and adds it to a graph.
    Inputs:
    info(dict): A dictionary of information about the entity.
    Returns:
    graph(graph): the graph with the added entity.
    """
    label = uri_to_label(info["@id"])  # Turning a URI into a label for an entity with @id(unique id for JASON-ID)
    safe_label = label.replace("'","\\'")  # A safe label to handle the replacement of single quotes
    code = f"MERGE (n:Entity {{name: '{safe_label}'}})"  # Cyper query to merge an entity wit the given label into the graph
    if 'multi-model' in info:
        for key in ['audio', 'image', 'visual']:
            if key in info['multi-model']:
               value = json.dumps(info['multi-model'][key])  # converting the key of the value into a JASON formatted string.
               safe_value = value.replace("'", "\\'")
               code += f" SET n.{key} = '{safe_value}'"


def add_relation(info, graph, start=None,end=None):
    """ Takes an entity and add a relation to the graph between entities.
    Inputs:
    info(dict): A dictionary of information about the entity.
    Returns:
    graph(graph): the graph with the relationships between the entity and a different entity.
    """
    end = info['end']['label']  # Create a list of  end nodes' labels
    surfaceText = info['surfaceText']

    if surfaceText:
        surfaceText = surfaceText.replace("'", "\\'")
    weight = info['weight']
    license = info['license']
    if license:
        license = license.replace("'", "\\'")
    graph.run('MERGE (n:Entity {name: $name})', name=start)  # Cyper Query to merge an entity to a start node
    graph.run('MERGE (n:Entity {name: $name})', name=end)  # Cypher Query to merge an entity to an end node

    code = '''MATCH (s:Entity {name: $start}),(t:Entity {name: $end}) '\
           MERGE (s)-[n:%s-{surfaceText:$surfaceText,weight: $weight,license: $license}]->(t)''' % label  # Cypher Query
    # The Match query looks for 2 nodes in the graph, the first node s is an Entity with a name property matching the value of start
    # The second node, t, is an Entity with a name property matching the value of end
    # The Merge query creates a relationship between 2 nodes. The relationship goes from node s to mode 2, and it has a type matching the value of lanel. Other properties set to the corresponding values.

    params = {
        'start': start,
        'end': end,
        'surfaceText': surfaceText,
        'weight': weight,
        'license': license
    }
    # Add properties for each modality in 'multi-model'
    for modality in ['audio','image','visual']:
        if modality in info.get['multi-model',{}]:
            data = json.dumps(info['multi-model'][modality])  # Converting objects into JSON strings
            code += 'SET n.%s = $%s' % (modality,modality)
            params[modality] = data

    graph.run(code , **params)

    

import os

os.system('python -m rdfizer -c ./config.ini')

import rdflib

g = rdflib.Graph()
res = g.parse("triples.nt", format="ntriples")

output = res.serialize(format='turtle')
print(output)


# What is the name of the personality that practises tennis?
res = g.query(
    """PREFIX ns1: <http://xmlns.com/foaf/0.1/> 
       PREFIX ns2: <http://example.com/ontology/>     
    
        SELECT ?who
        WHERE { ?x ns1:name ?who .
                ?x ns2:practises ?y .
                ?y rdfs:label "Tennis" .
               } 
    """)

print(res.serialize(format="json"))

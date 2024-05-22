CLI usage

./cli.sh query 
example:
./cli.sh query "How does the Neuron class work?"
- General query about the code, ask about specific files or codeblocks 
- optional arguments  
-- code directory, if directory other than uppgift_2 is wanted, enter directory path as the first argument
-- create new embeddings, if directory already has embeddings but you wish to generate new embeddings, write true as the second argument
-- example:
-- ./cli.sh query "How does the Neuron class work?" ./Uppgift_2 true

./cli.sh 
example: 
./cli.sh summary "/Uppgift_2/A.py"
- summarizes specific file or all files in specific directory.


./cli.sh modify 
example:
./cli.sh modify "/Uppgift_2/A.py" "Add more comments to the Neuron class"
- Gives code improvements for specific file according to description given. 
- Offers to modify file with new code
- If user agrees the code will be changed and pushed to github
-- code repository needs to be inside of git repo to work properly
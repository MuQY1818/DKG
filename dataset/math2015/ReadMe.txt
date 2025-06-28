1. There are three directories in the current folder, and each directory contains a piece of data used in our paper as follows:
FrcSub-----------------The public dataset, widely used in cognitive modelling (e.g., [Tatsuoka, 1984; Junker and Sijtsma, 2001; DeCarlo, 2010]), is made up of test responses (right or wrong, coded to 1 or 0) of examinees on Fraction-Substraction problems.
Math1&Math2------------The private datasets we used include two final math examination results (scores of each examinee on each problem) of a high school.

2. There are four files in each directory as follows:
data.txt---------------The responses or normalized scores (which are scaled in range [0,1] by dividing full scores of each problem) of each examinee on each problems, and a row denotes an examinee while a column stands for a problem.
qnames.txt-------------The detailed names or meanings of related specific skill.
q.txt------------------The indicator matrix of relationship between problems and skills, which derives from experienced education experts. And a row represents a problem while a column for a skill. E.g., problem i requires skill k if entry(i, k) equals to 1 and vice versa.
problemdesc.txt--------The description of each problem, including the problem type (objective or subjective) and full scores of each problem (set to 1 for all the problems in FrcSub dataset).

3. Besides, there is one more file in Math1 and Math2 directories.
rawdata.txt------------The raw unnormalized scores of the Math1 and Math2 datasets.

4. For better understanding, we give two examples of how to use the datasets in the file "Example.txt" in the current folder.

5. And if you intend to use the two private datasets (called Math dataset) for any exploratory analysis, please refer to the Terms of Use, which is decribed in the file "TermsOfUse.txt" in detail.


1. 当前文件夹中包含三个目录，每个目录中都包含一篇论文中使用的数据，具体如下：
FrcSub-----------------该公开数据集在认知建模领域广泛应用（例如[Tatsuoka, 1984; Junker and Sijtsma, 2001; DeCarlo, 2010]），包含受试者在分数减法问题测试中的答题结果（正确或错误，分别编码为1或0）。
Math1&Math2------------我们使用的私有数据集包括一所高中的两次数学期末考试结果（每位受试者在每个问题上的得分）。

2. 每个目录下包含四个文件，具体如下：
data.txt---------------每位考生在每个问题上的回答或标准化分数（通过将每个问题的满分除以100，将分数缩放至[0,1]范围），其中一行代表一位考生，一列代表一个问题。
qnames.txt-------------与特定技能相关的详细名称或含义。
q.txt------------------问题与技能之间关系指标矩阵，由资深教育专家制定。其中，行代表问题，列代表技能。例如，若问题i需要技能k，则(i, k)单元格等于1，反之亦然。
problemdesc.txt--------每个问题的描述，包括问题类型（客观题或主观题）及每个问题的满分（FrcSub数据集中的所有问题均设为1分）。

3. 此外，Math1和Math2目录中还各有一个文件。
rawdata.txt------------Math1和Math2数据集的原始未标准化分数。

4. 为了更好地理解，我们在当前文件夹中的 “Example.txt” 文件中提供了两个使用数据集的示例。

5. 如果您打算使用这两个私有数据集（称为 Math 数据集）进行任何探索性分析，请参阅使用条款，该条款在 “TermsOfUse.txt” 文件中详细描述。

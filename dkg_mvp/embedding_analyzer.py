import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os

class EmbeddingAnalyzer:
    """
    分析和利用节点嵌入的工具类。
    """
    def __init__(self, embeddings_path='models/embeddings'):
        """
        初始化分析器并加载嵌入。
        
        Args:
            embeddings_path (str): 存放嵌入文件的目录路径。
        """
        self.embeddings_path = embeddings_path
        self.problem_embeddings = self._load_embeddings('problem_embeddings.csv')
        self.skill_embeddings = self._load_embeddings('skill_embeddings.csv')
        
        if self.problem_embeddings is None:
            raise FileNotFoundError("Problem embeddings could not be loaded. Please run gnn_trainer.py first.")

    def _load_embeddings(self, filename: str) -> pd.DataFrame:
        """加载指定的嵌入文件。"""
        file_path = os.path.join(self.embeddings_path, filename)
        if not os.path.exists(file_path):
            print(f"Warning: Embedding file not found at {file_path}")
            return None
        print(f"Loading embeddings from {file_path}...")
        df = pd.read_csv(file_path, index_col=0)
        # 确保索引是整数类型
        df.index = df.index.astype(int)
        return df

    def find_similar_problems(self, problem_id: int, top_n: int = 5):
        """
        根据嵌入向量的余弦相似度，查找与给定问题最相似的问题。

        Args:
            problem_id (int): 要查询的目标问题ID。
            top_n (int): 要返回的最相似问题的数量。

        Returns:
            pd.DataFrame: 包含最相似问题ID及其相似度分数的数据帧，按相似度降序排列。
        """
        if problem_id not in self.problem_embeddings.index:
            return f"Error: Problem ID {problem_id} not found in embeddings."

        # 获取目标问题的嵌入向量
        target_vec = self.problem_embeddings.loc[problem_id].values.reshape(1, -1)

        # 计算余弦相似度
        sim_scores = cosine_similarity(target_vec, self.problem_embeddings.values)

        # 将相似度分数与问题ID关联
        sim_df = pd.DataFrame(sim_scores.T, index=self.problem_embeddings.index, columns=['similarity'])
        
        # 移除自身
        sim_df = sim_df.drop(problem_id)
        
        # 按相似度降序排序并返回top_n
        most_similar = sim_df.sort_values(by='similarity', ascending=False).head(top_n)
        
        return most_similar

    def find_similar_skills(self, skill_id: int, top_n: int = 5):
        """
        根据嵌入向量的余弦相似度，查找与给定技能最相似的技能。

        Args:
            skill_id (int): 要查询的目标技能ID。
            top_n (int): 要返回的最相似技能的数量。

        Returns:
            pd.DataFrame: 包含最相似技能ID及其相似度分数的数据帧。
        """
        if self.skill_embeddings is None:
            return "Skill embeddings are not loaded."
            
        if skill_id not in self.skill_embeddings.index:
            return f"Error: Skill ID {skill_id} not found in embeddings."

        target_vec = self.skill_embeddings.loc[skill_id].values.reshape(1, -1)
        sim_scores = cosine_similarity(target_vec, self.skill_embeddings.values)
        sim_df = pd.DataFrame(sim_scores.T, index=self.skill_embeddings.index, columns=['similarity'])
        sim_df = sim_df.drop(skill_id)
        most_similar = sim_df.sort_values(by='similarity', ascending=False).head(top_n)
        
        return most_similar


def main():
    """主执行函数，用于命令行交互。"""
    parser = argparse.ArgumentParser(description="Find similar items based on GNN embeddings.")
    parser.add_argument('--problem_id', type=int, help="The ID of the problem to find similarities for.")
    parser.add_argument('--skill_id', type=int, help="The ID of the skill to find similarities for.")
    parser.add_argument(
        '--top_n', 
        type=int, 
        default=5, 
        help="Number of similar items to return."
    )
    args = parser.parse_args()

    if not args.problem_id and not args.skill_id:
        parser.error("Please provide either --problem_id or --skill_id.")

    try:
        analyzer = EmbeddingAnalyzer()
        
        if args.problem_id:
            problem_id_to_query = args.problem_id
            similar_items = analyzer.find_similar_problems(problem_id_to_query, args.top_n)
            item_type = 'problems'
            item_id = problem_id_to_query
        
        elif args.skill_id:
            skill_id_to_query = args.skill_id
            similar_items = analyzer.find_similar_skills(skill_id_to_query, args.top_n)
            item_type = 'skills'
            item_id = skill_id_to_query

        print(f"--- Top {args.top_n} {item_type} most similar to {item_type[:-1]} '{item_id}' ---")
        print(similar_items)
        print("\n")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main() 
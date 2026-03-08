from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:

    def __init__(self,threshold=0.85):

        self.cache=[]

        self.threshold=threshold

        self.hit_count=0

        self.miss_count=0

        self.model=SentenceTransformer("all-MiniLM-L6-v2")


    def search(self,query):

        query_embedding=self.model.encode([query])

        for entry in self.cache:

            sim=cosine_similarity(query_embedding,entry["embedding"])[0][0]

            if sim>=self.threshold:

                self.hit_count+=1

                return True,entry,sim

        self.miss_count+=1

        return False,None,None


    def add(self,query,result):

        embedding=self.model.encode([query])

        self.cache.append({
            "query":query,
            "embedding":embedding,
            "result":result
        })


    def stats(self):

        total=len(self.cache)

        hit_rate=self.hit_count/(self.hit_count+self.miss_count+1e-5)

        return {
            "total_entries":total,
            "hit_count":self.hit_count,
            "miss_count":self.miss_count,
            "hit_rate":hit_rate
        }


    def clear(self):

        self.cache=[]

        self.hit_count=0

        self.miss_count=0
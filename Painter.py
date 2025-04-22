import matplotlib.pyplot as plt

class Draw:
    def __init__(self, path, title="", legend=False):
        self.path=path
        self.title=title
        self.legend=legend
    
    def __enter__(self):
        plt.figure(figsize=(6, 6)) 
        plt.title(self.title)
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        if(self.legend):
            plt.legend()
        plt.savefig(self.path)
        plt.close()

class DrawROC:
    def __init__(self, path, title="", legend=False):
        self.path=path
        self.title=title
        self.legend=legend
    
    def __enter__(self):
        plt.figure(figsize=(6, 6)) 
        plt.title(self.title)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        if(self.legend):
            plt.legend()
        plt.savefig(self.path)
        plt.close()



from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

@dataclass
class Column:
    name: str
    concept: str
    encoding: str
    eda_reg_r2: float

class ColumnMetaOrganiser:
    def __init__(self):
        self.columns: List[Column] = []

    def add_col(self, column: Column):
        self.columns.append(column)
    
    def select_cols(self, concepts: List[str], encodings: List[str], include_cols= [], exclude_cols=[]) -> List[str]:
        selected = [
            c.name for c in self.columns
            if c.concept in concepts 
            and c.encoding in encodings
            and c.name not in exclude_cols
        ]
        selected.extend([col for col in include_cols if col not in selected])
        return selected
    
    def filter_df(self, df, cols: List[str]):
        return df.loc[:, cols]

    def summarise_concepts(self) -> pd.DataFrame:
        rows: List[Tuple[str, str, int, float]] = []
        for (concept, encoding), group in self._group_by_concept_encoding().items():
            count = len(group)
            avg_r2 = sum(c.eda_reg_r2 for c in group) / count
            rows.append((concept, encoding, count, avg_r2))
        summary_df = pd.DataFrame(rows, columns=["concept", "encoding", "n_cols", "avg_r2"])
        return summary_df.sort_values(by="avg_r2", ascending=False).reset_index(drop=True)

    def _group_by_concept_encoding(self):
        grouped = {}
        for col in self.columns:
            key = (col.concept, col.encoding)
            grouped.setdefault(key, []).append(col)
        return grouped
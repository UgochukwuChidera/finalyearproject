import csv
import json
from pathlib import Path

class DataExporter:
    def export_json(self, structured, output_path):
        path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("w",encoding="utf-8") as fh:
            json.dump(structured,fh,indent=2,default=str,ensure_ascii=False)
        return str(path)

    def export_csv(self, structured, output_path):
        path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("w",newline="",encoding="utf-8") as fh:
            w=csv.writer(fh)
            w.writerow(["field","value"])
            w.writerow(["_form_id",structured.get("form_id","")])
            w.writerow(["_template_id",structured.get("template_id","")])
            w.writerow(["_processed_at",structured.get("processed_at","")])
            for k,v in structured.get("data",{}).items():
                w.writerow([k,v])
        return str(path)

    def export_all(self, structured, base_path):
        base=str(Path(base_path).with_suffix(""))
        return {"json":self.export_json(structured,base+".json"),
                "csv":self.export_csv(structured,base+".csv")}

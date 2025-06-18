from sqlalchemy import and_

from mimic.orm_create.mimiciv_v3_orm import DLabitems


class LabItemProp:
    def __init__(self, session, prop):
        self.session = session
        self.prop = prop

    def _get_wbc_item(self):
        return self.session.query(self.prop).filter(
            and_(DLabitems.label == "White Blood Cells", DLabitems.category == "Hematology")).distinct().first()[0]

    def _get_haemoglobin_item(self):
        """Finds distinct lab items related to haemoglobin."""
        return self.session.query(self.prop).filter(
            and_(DLabitems.label == "Hemoglobin", DLabitems.category == "Hematology")).distinct().first()[0]

    def _get_platelets_item(self):
        """Finds distinct lab items related to platelets."""
        return self.session.query(self.prop).filter(
            and_(DLabitems.label == "Platelet Count", DLabitems.category == "Hematology")).distinct().first()[0]

    def _get_mean_corpuscular_volume_item(self):
        """Finds distinct lab items related to mean corpuscular volume."""
        return self.session.query(self.prop).filter(
            and_(DLabitems.label == "MCV", DLabitems.category == "Hematology")).distinct().first()[0]

    def _get_red_blood_cells_item(self):
        """Finds distinct lab items related to both white and red blood cells."""
        return self.session.query(self.prop).filter(
            and_(DLabitems.label == "Red Blood Cells", DLabitems.category == "Hematology")
        ).distinct().first()[0]

    # def _get_crp_item(self):
    #     """Finds distinct lab items related to both white and red blood cells."""
    #     return self.session.query(self.prop).filter(
    #         and_(DLabitems.label == "C-Reactive Protein")).distinct().first()[0]

    def get_all_method_results(self):
        results = {}
        for name in dir(self):
            if not name.startswith('_') or name.startswith("__"): continue
            attribute = getattr(self, name)
            if callable(attribute):
                results[name] = attribute()
        return results
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson.objectid import ObjectId
from datetime import datetime


# Usage:
# log_client = MongoClient("mongodb://127.0.0.1:27017")
# logger = RunInfoRegression(log_client, 'us_leasing_exploration', 'runs')
# logger.save('All features only', {'rmse': 0.5})
# logger.delete({'_id': ObjectId('5a0597b98ce3de1663b8e61c')})
# cursor = logger.get(sort=[('time', ASCENDING)])
#

# for document in cursor:
#     print(document)
# cursor = log_client['us_leasing_exploration']['runs'].find()
# .get({"borough": "Manhattan"})
# .get({"grades.grade": "B"})
# .get({"grades.score": {"$gt": 30}})
# .get({"cuisine": "Italian", "address.zipcode": "10075"})
#      {"$or": [{"cuisine": "Italian"}, {"address.zipcode": "10075"}]})
# .sort([
#     ("borough", ASCENDING),
#     ("address.zipcode", ASCENDING)
# ])


class RunInfo:
    def __init__(self, client, db_name, table_name):
        self.client = client
        self.db_name = db_name
        self.table_name = table_name

    def save(self, model_type, comment, metrics,
             features_importance={}, features=[], model_params={}, run_time=None,
             model_description=None, etc=None):
        db = self.client[self.db_name]
        collection = db[self.table_name]

        now = datetime.now() if run_time is None else run_time

        result = collection.insert_one({
            'time': now,
            'comment': comment,
            'metrics': metrics,
            'model_type': model_type,
            'model_description': model_description,
            'features': features,
            'features_importance': features_importance,
            'model_params': model_params,
            'etc': etc
        })

        return result.inserted_id

    def get(self, condition=None, sort=[("time", DESCENDING)]):
        return self.client[self.db_name][self.table_name].find(condition).sort(sort)

    def delete(self, condition):
        return self.client[self.db_name][self.table_name].delete_many(condition)


class RunInfoRegression(RunInfo):
    def save(self, comment, metrics,
             features_importance={}, features=[], model_params={}, run_time=None,
             model_description=None, etc=None):

        super(RunInfoRegression, self).save('regression', comment, metrics,
                                            features_importance=features_importance, features=features,
                                            model_params=model_params,
                                            run_time=run_time, model_description=model_description, etc=etc)

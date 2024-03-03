from motor.motor_asyncio import AsyncIOMotorClient

MONGO_DETAILS = "mongodb+srv://gordenggm:Uzuv4l9Ogs3AsnFE@cluster0.k9lwx62.mongodb.net/"
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.NLP
question_collection = database.QA
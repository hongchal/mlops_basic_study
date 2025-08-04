from fastapi import FastAPI 
from pydantic import BaseModel 

app = FastAPI()

# database
ITEMS = {0: "default data"}

# create

class ItemCreateIn(BaseModel):
    item_body: str 

class ItemCreateOut(BaseModel):
    item_id: int 
    item_body: str

@app.post("/item/")
def create_item(item_create_in: ItemCreateIn) -> ItemCreateOut:
    item_id = len(ITEMS)
    item_body = item_create_in.item_body
    ITEMS[item_id] = item_body
    return ItemCreateOut(item_id=item_id, item_body=item_body)

# read 

class ItemGetOut(BaseModel):
    item_id: int 
    item_body: str
 
@app.get("/item/", response_model=ItemGetOut)
def read_item(item_id: int) -> ItemGetOut:
    item_body = ITEMS.get(item_id, "Not valid id")
    return ItemGetOut(item_id=item_id, item_body=item_body)


# update 

class ItemUpdateIn(BaseModel):
    item_id: int 
    item_body: str 

class ItemUpdateOut(BaseModel):
    item_id: int 
    item_body: str

@app.put("/item/", response_model=ItemUpdateOut)
def update_item(item_update_in: ItemUpdateIn) -> ItemUpdateOut:
    item_id = item_update_in.item_id
    item_body = item_update_in.item_body

    if item_id not in ITEMS:
        item_body = "Not Valid"
    else:
        ITEMS[item_id] = item_body 
    return ItemUpdateOut(item_id=item_id, item_body=item_body)

# delete
# @app.delete
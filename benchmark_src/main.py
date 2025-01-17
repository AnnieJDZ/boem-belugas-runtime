from pathlib import Path
import random
from loguru import logger
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
#import torchvision.models as models
from resnet import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from label_generation import *

ROOT_DIRECTORY = Path("C:/Users/zpmao/Downloads/boem-belugas-runtime") #Path("/code_execution")
PREDICTION_FILE = ROOT_DIRECTORY / "submission" / "submission.csv"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

pretrained_model_name = "resnet34-333f7ec4.pth"

# BENCHMARK_SRC = ROOT_DIRECTORY / "benchmark_src"
# model = resnet34()
# model.load_state_dict("resnet34-333f7ec4.pth") #models.resnet34(pretrained=True)
# torch.save(model, BENCHMARK_SRC / "model.pth")

class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id and image tensors.
    """

    def __init__(self, metadata):
        self.metadata = metadata
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __getitem__(self, idx):
        image = Image.open(DATA_DIRECTORY / self.metadata.path.iloc[idx]).convert("RGB")
        image = self.transform(image)
        sample = {"image_id": self.metadata.index[idx], "image": image}
        return sample

    def __len__(self):
        return len(self.metadata)


def main():
    logger.info("Starting main script")
    # load test set data and pretrained model
    query_scenarios = pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv", index_col="scenario_id")
    metadata = pd.read_csv(DATA_DIRECTORY / "metadata.csv", index_col="image_id")
    logger.info("Loading pre-trained model")
    model = resnet34()
    model.load_state_dict(torch.load(pretrained_model_name))
    # model = torch.load("model.pth")

    # we'll only precompute embeddings for the images in the scenario files (rather than all images), so that the
    # benchmark example can run quickly when doing local testing. this subsetting step is not necessary for an actual
    # code submission since all the images in the test environment metadata also belong to a query or database.
    scenario_imgs = []
    for row in query_scenarios.itertuples():
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.queries_path).query_image_id.values)
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values)

    # No need to sort yet, we make these a set in the end

    # First we shuffle scenario  images
    random.shuffle(scenario_imgs)
    print(len(scenario_imgs))

    split_ratio = 0.7
    split_index = int(split_ratio*len(scenario_imgs))
    train_scen_imgs = scenario_imgs[:split_index]
    val_scen_imgs = scenario_imgs[split_index:]

    # Make these sets
    train_scen_imgs = sorted(set(train_scen_imgs))
    val_scen_imgs = sorted(set(val_scen_imgs))
    full_dataset = sorted(set(scenario_imgs))

    metadata = metadata.loc[full_dataset]
    train_metadata = metadata.loc[train_scen_imgs]
    val_metadata = metadata.loc[val_scen_imgs]

    # instantiate dataset/loader and generate embeddings for all images
    dataset = ImagesDataset(metadata)
    dataloader = DataLoader(dataset, batch_size=16)
    train_dataloader = DataLoader(train_metadata, batch_size=16)
    val_dataloader = DataLoader(val_metadata, batch_size=16)
    
    embeddings = []
    model.eval()

    logger.info("Precomputing embeddings")
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch_embeddings = model(batch["image"])
        batch_embeddings_df = pd.DataFrame(batch_embeddings.detach().numpy(), index=batch["image_id"])
        embeddings.append(batch_embeddings_df)

    embeddings = pd.concat(embeddings)
    logger.info(f"Precomputed embeddings for {len(embeddings)} images")

    logger.info("Generating image rankings")
    # process all scenarios
    results = []
    for row in query_scenarios.itertuples():
        # load query df and database images; subset embeddings to this scenario's database
        qry_df = pd.read_csv(DATA_DIRECTORY / row.queries_path)
        db_img_ids = pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values
        db_embeddings = embeddings.loc[db_img_ids]

        # predict matches for each query in this scenario
        for qry in qry_df.itertuples():
            # get embeddings; drop query from database, if it exists
            qry_embedding = embeddings.loc[[qry.query_image_id]]
            _db_embeddings = db_embeddings.drop(qry.query_image_id, errors='ignore')

            # compute cosine similarities and get top 20
            sims = cosine_similarity(qry_embedding, _db_embeddings)[0]
            top20 = pd.Series(sims, index=_db_embeddings.index).sort_values(0, ascending=False).head(20)

            # append result
            qry_result = pd.DataFrame(
                {"query_id": qry.query_id, "database_image_id": top20.index, "score": top20.values}
            )
            results.append(qry_result)

    logger.info(f"Writing predictions file to {PREDICTION_FILE}")
    submission = pd.concat(results)
    submission.to_csv(PREDICTION_FILE, index=False)


if __name__ == "__main__":
    main()

# trainer for gpt model
import tiktoken
import torch
import time

from models.gpt2.dataset import create_dataloader
from models.gpt2.model import GPT
from loguru import logger


logger.add("gpt2.log", rotation="10 MB", retention="10 days", compression="zip")


def get_dataloader(dataset, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)

    # target_batch = torch.unsqueeze(target_batch, -1)
    # loss = torch.nn.functional.cross_entropy(logits, target_batch)
    # add dim 1 to target_batch
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


# eval model
def eval_model(model, val_loader, device):
    """
    eval model 
    :param model: model to eval
    :param val_loader: validation data loader
    :param device: device to run model on
    :return: avg_loss, avg_perplexity 
    """
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_perplexity = 0
        for input_batch, target_batch in val_loader:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            total_perplexity += torch.exp(loss).item()
        avg_loss = total_loss / len(val_loader)
        avg_perplexity = total_perplexity / len(val_loader)
        return avg_loss, avg_perplexity


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """

    :param model:
    :param train_loader:
    :param val_loader:
    :param optimizer:
    :param device:
    :param num_epochs:
    :param eval_freq:
    :param eval_iter:
    :param start_context:
    :param tokenizer:
    :return:
    """

    train_losses, val_losses, track_tokens_seen = [], [], []
    best_perplexity = float('inf')  # Initialize best_perplexity with infinity

    # run model over num_epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        last_time = start_time
        for iteration, (input_batch, target_batch) in enumerate(train_loader):
            try:
                # reset optimizer
                optimizer.zero_grad()

                # calculate loss
                loss = calc_loss_batch(input_batch, target_batch, model, device)

                # calculate perplexity
                perplexity = torch.exp(loss)

                loss.backward()
                optimizer.step()

                # Calculate progress percentage
                progress = (iteration + 1) / len(train_loader) * 100

                logger.info(f"Epoch {epoch + 1}, Iteration {iteration + 1} ({progress:.2f}%), Train Loss: {loss.item()}, TrainPerplexity: {perplexity.item()}")
                
                
                #########################################################
                # eval model
                if (iteration + 1) % 200 == 0:
                    model.eval()
                    with torch.no_grad():

                        # Calculate eval loss
                        # eval_loss = calc_loss_batch(input_batch, target_batch, model, device)
                        # logger.info(f"Epoch {epoch + 1}, Iteration {iteration + 1} ({progress:.2f}%), Eval Loss: {eval_loss.item()}")

                        eval_loss, eval_perplexity = eval_model(model, val_loader, device)
                        logger.info(f"Epoch {epoch + 1}, Iteration {iteration + 1} ({progress:.2f}%), Eval Loss: {eval_loss}, Eval Perplexity: {eval_perplexity}")

                        # Save the model after training
                        torch.save(model.state_dict(), 'last.pt')
                        logger.info("Last model saved successfully.")

                        # Save the model with best perplexity
                        if eval_perplexity < best_perplexity:
                            best_perplexity = eval_perplexity
                            torch.save(model.state_dict(), 'best.pt')
                            logger.info("Best model saved successfully.")

                        context = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0).to(device)
                        generated = context
                        for _ in range(50):  # Generate 50 new tokens
                            outputs = model(generated)
                            next_token = outputs[:, -1, :].argmax(dim=-1).unsqueeze(-1)
                            generated = torch.cat([generated, next_token], dim=-1)
                        generated_text = tokenizer.decode(generated[0].tolist())
                        track_tokens_seen.append(generated_text)
                        logger.info(f"Epoch {epoch + 1}, Iteration {iteration + 1} ({progress:.2f}%), Generated Text: {generated_text}")

                    model.train()

                # Display train speed in real time
                current_time = time.time()
                elapsed_time = current_time - last_time
                iterations_per_second = 1 / elapsed_time
                logger.info(f"Epoch {epoch + 1}, Iteration {iteration + 1}, Speed: {iterations_per_second:.2f} iterations per second")
                last_time = current_time
            except Exception as e:
                logger.info(f"Error: {e}")
        end_time = time.time()
        epoch_time = end_time - start_time
        logger.info(f"Epoch {epoch + 1} execution time: {epoch_time} seconds")
        logger.info(f"Epoch {epoch + 1} speed: {len(train_loader) / epoch_time} iterations per second")


    # return losses
    # return loss


def main(GPT_CONFIG_124M, OTHER_SETTINGS):
    text_data = ""
    # read local file content the-verdict.txt
    with open("./OpenWikiText10k.txt", "r", encoding='utf-8') as f:
        text_data = f.read()

    model = GPT(GPT_CONFIG_124M)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # adamw optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=OTHER_SETTINGS["learning_rate"],
                                  weight_decay=OTHER_SETTINGS["weight_decay"])

    ##############################
    # Set up dataloaders
    ##############################
    # split text_data into train, val by ratio 0.9  using create_dataloader_v1
    train_dataloader = create_dataloader(text_data, batch_size=OTHER_SETTINGS["batch_size"],
                                         max_length=GPT_CONFIG_124M["context_length"],
                                         stride=GPT_CONFIG_124M["context_length"] // 2)

    val_dataloader = create_dataloader(text_data, batch_size=OTHER_SETTINGS["batch_size"],
                                       max_length=GPT_CONFIG_124M["context_length"],
                                       stride=GPT_CONFIG_124M["context_length"] // 2)

    # logger.info train dataset basic information
    train_dataset_size = len(train_dataloader.dataset)
    num_iterations = len(train_dataloader)
    logger.info(f"Train dataset size: {train_dataset_size}")
    logger.info(f"Number of iterations per epoch for train: {num_iterations}")

    # logger.info val dataset basic information
    val_dataset_size = len(val_dataloader.dataset)
    num_iterations_val = len(val_dataloader)
    logger.info(f"Eval dataset size: {val_dataset_size}")
    logger.info(f"Number of iterations per epoch for validation: {num_iterations_val}")
    

    ##############################
    # Train model
    ##############################
    # tiktoken tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    train_model_simple(model, train_dataloader, val_dataloader, optimizer, device, num_epochs=OTHER_SETTINGS["num_epochs"], eval_freq=5, eval_iter=1,
                       start_context="I like eating apple", tokenizer=tokenizer)


if __name__ == '__main__':
    # GPT_CONFIG_124M
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "emb_dim": 768,
        "context_length": 1024,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    # other settings
    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 1,
        "batch_size": 4,
        "weight_decay": 0.1
    }

    main(GPT_CONFIG_124M, OTHER_SETTINGS)

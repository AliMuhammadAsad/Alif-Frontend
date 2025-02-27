# Model/data_loader.py
import torch
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextDataLoader:
    def __init__(self, file_path, batch_size, block_size, tokenizer, chunk_size=10**4):
        self.file_path = file_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.file = open(self.file_path, 'r', encoding='utf-8')
        self.data = None
        self.end_of_file = False

        # Load the initial chunk of data
        self.load_chunk()

    def load_chunk(self):
        """Load a chunk from the file, encode it, and handle end-of-file conditions."""
        text = self.file.read()
        if not text:
            self.end_of_file = True
            logging.info("End of file reached.")
        else:
            try:
                # Encode the text using the tokenizer
                encoded = self.tokenizer.encode(text)
                if len(encoded) > 0:
                    self.data = torch.tensor(encoded, dtype=torch.long)
                    logging.info(f"Loaded new data chunk of size: {len(self.data)} tokens.")
                    # save the encoded data to a file
                    torch.save(self.data, "encoded_data.pth")
            except Exception as e:
                logging.error(f"Error encoding text chunk: {e}")
                self.end_of_file = True

    def num_batches(self):
        """Calculate the total number of batches in the current chunk."""
        if self.data is not None:
            return (len(self.data) - 1) // self.block_size  # Total batches in the current chunk
        return 0

    def get_batch(self):
        """Retrieve a batch of data from the current chunk or load a new chunk if needed."""
        if self.end_of_file:
            return None, None  # Return None when no data is left

        # Generate a batch of data
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x, y

    def reset(self):
        """Reset the file and flags for a new epoch."""
        self.file.seek(0)
        self.end_of_file = False
        logging.info("Resetting file for a new epoch.")
        self.load_chunk()

    def close(self):
        """Clean up file resources when done."""
        self.file.close()
        logging.info("File closed.")

    def __iter__(self):
        """Make the data loader iterable so it can be used in a loop."""
        while not self.end_of_file:
            x, y = self.get_batch()
            if x is None or y is None:
                break  # Stop iteration if there's no more data

            yield x, y  # Yield a batch of data

        # Once iteration is done, close the file
        self.close()

#before parallelizing
# Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class TextDataLoader:
#     def __init__(self, file_path, batch_size, block_size, tokenizer, device='cpu', chunk_size=10**4):
#         self.file_path = file_path
#         self.batch_size = batch_size
#         self.block_size = block_size
#         self.tokenizer = tokenizer
#         self.device = device
#         self.chunk_size = chunk_size
#         self.file = open(self.file_path, 'r', encoding='utf-8')
#         self.data = None
#         self.end_of_file = False

#         # Load the initial chunk of data
#         self.load_chunk()

#     def load_chunk(self):
#         """Load a chunk from the file, encode it, and handle end-of-file conditions."""
#         text = self.file.read()
#         if not text:
#             self.end_of_file = True
#             logging.info("End of file reached.")
#         else:
#             try:
#                 # Encode the text using the tokenizer
#                 encoded = self.tokenizer.encode(text)
#                 if len(encoded) > 0:
#                     self.data = torch.tensor(encoded, dtype=torch.long).to(self.device)
#                     logging.info(f"Loaded new data chunk of size: {len(self.data)} tokens.")
#             except Exception as e:
#                 logging.error(f"Error encoding text chunk: {e}")
#                 self.end_of_file = True

#     def num_batches(self):
#         """Calculate the total number of batches in the current chunk."""
#         if self.data is not None:
#             return (len(self.data) - 1) // self.block_size  # Total batches in the current chunk
#         return 0

#     def get_batch(self):
#         """Retrieve a batch of data from the current chunk or load a new chunk if needed."""
#         if self.end_of_file:
#             return None, None  # Return None when no data is left
        
#         # Generate a batch of data
#         ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
#         x = torch.stack([self.data[i:i+self.block_size] for i in ix])
#         y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
#         return x, y

#     def reset(self):
#         """Reset the file and flags for a new epoch."""
#         self.file.seek(0)
#         self.end_of_file = False
#         logging.info("Resetting file for a new epoch.")
#         self.load_chunk()

#     def close(self):
#         """Clean up file resources when done."""
#         self.file.close()
#         logging.info("File closed.")

#     def __iter__(self):
#         """Make the data loader iterable so it can be used in a loop."""
#         while not self.end_of_file:
#             x, y = self.get_batch()
#             if x is None or y is None:
#                 break  # Stop iteration if there's no more data
            
#             yield x, y  # Yield a batch of data

#         # Once iteration is done, close the file
#         self.close()


# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class TextDataLoader:
#     def __init__(self, file_path, batch_size, block_size, tokenizer, device='cpu', chunk_size=10**4):
#         self.file_path = file_path
#         self.batch_size = batch_size
#         self.block_size = block_size
#         self.tokenizer = tokenizer
#         self.device = device
#         self.chunk_size = chunk_size
#         self.file = open(self.file_path, 'r', encoding='utf-8')
#         self.data = None
#         self.end_of_file = False

#         # Print a preview of the file
#         # self.print_file_preview()
        
#         # Initial chunk loading
#         self.load_chunk()

#     def print_file_preview(self):
#         """Prints the first few lines of the text file for preview"""
#         self.file.seek(0)  # Go to the beginning of the file
#         lines = [self.file.readline() for _ in range(5)]
#         preview_text = ''.join(lines)
#         print("File preview:\n", preview_text)
#         self.file.seek(0)  # Reset to the start of the file for chunk reading

#     def load_chunk(self):
#         """Load a chunk from the file, encode it, and handle end-of-file conditions."""
#         text = self.file.read()
#         if not text:
#             self.end_of_file = True
#             logging.info("End of file reached.")
#         else:
#             try:
#                 # Log the first 100 characters of the text chunk to verify Urdu content
#                 # logging.info(f"First 100 characters of the chunk: {text[:100]}")
#                 # print("This is the chunk:", text)

#                 # Encode the text using the tokenizer
#                 # print("Tokenizer:", self.tokenizer)
#                 encoded = self.tokenizer.encode(text)
#                 print(len(encoded))
#                 print("encoded data: ")

#                 # Log the encoded output length to confirm successful encoding
#                 logging.info(f"Encoded data length: {len(encoded)} tokens")

#                 # if len(encoded) < self.block_size:
#                 #     # Only stop if there's absolutely no usable data left
#                 #     self.end_of_file = len(encoded) == 0
#                 #     if self.end_of_file:
#                 #         logging.warning("Insufficient data in chunk; stopping further loading.")
#                 #     else:
#                 #         logging.warning("Data chunk smaller than block size loaded; may limit training batch size.")
                
#                 if len(encoded) > 0:
#                     self.data = torch.tensor(encoded, dtype=torch.long).to(self.device)
#                     logging.info(f"Loaded new data chunk of size: {len(self.data)} tokens.")
#             except Exception as e:
#                 logging.error(f"Error encoding text chunk: {e}")
#                 self.end_of_file = True

#     def get_batch(self):
#         """Retrieve a batch of data from the current chunk or load a new chunk if needed."""
#         # if self.end_of_file:
#         #     return None, None  # Return None when no data is left
        
#         # if self.data is None or len(self.data) <= self.block_size:
#         #     self.load_chunk()
#         #     if self.end_of_file or self.data is None or len(self.data) < self.block_size:
#         #         return None, None  # Stop if there’s insufficient data

#         # Generate a batch of data
#         ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
#         x = torch.stack([self.data[i:i+self.block_size] for i in ix])
#         y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
#         return x, y

#     def reset(self):
#         """Reset the file and flags for a new epoch."""
#         self.file.seek(0)
#         self.end_of_file = False
#         logging.info("Resetting file for a new epoch.")
#         self.load_chunk()

#     def close(self):
#         """Clean up file resources when done."""
#         self.file.close()
#         logging.info("File closed.")
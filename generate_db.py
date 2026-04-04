import pandas as pd
import numpy as np

def generate_data(num_rows=1_000_000):
    print(f"Generating {num_rows} rows of synthetic data...")

    num_outliers = 10
    num_normal = num_rows - num_outliers

    sensor_data = np.concatenate([
        np.random.randint(10, 20, num_normal), 
        np.random.randint(1000, 5000, num_outliers)
    ])
    assert len(sensor_data) == num_rows

    print(f"Sensor data: {sensor_data}")
    
    data = {
        # 1. Low-cardinality (for Bitmap Index)
        "status": np.random.choice(["Active", "Inactive", "Pending"], num_rows),
        
        # 2. Sorted/Clustered Integers (for RLE & Zone Maps)
        "cluster_id": np.sort(np.random.randint(0, 50, num_rows)),
        
        # 3. Uniform Random (The 'Hard' case for skipping)
        "random_val": np.random.randint(0, 1000, num_rows),
        
        # 4. Monotonic (for Delta + Bit Packing)
        "timestamp": np.arange(num_rows) + 1600000000,
        
        # 5. Outliers (for Mostly Encoding)
        "sensor_reading": sensor_data
    }
    
    # Shuffle only the rows that shouldn't be sorted
    df = pd.DataFrame(data)
    
    # Save for reproducibility as required by the Loader spec
    df.to_csv("synthetic_data.csv", index=False)
    print("Done! saved to synthetic_data.csv")

if __name__ == "__main__":
    generate_data()
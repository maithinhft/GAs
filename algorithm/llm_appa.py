import os
import json
import math
import numpy as np
from typing import List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

from algorithm.appa import APPAAlgorithm
from utils.config import UAV, Region

# Load environment variables
load_dotenv()

class LLMAPPAAlgorithm(APPAAlgorithm):
    def __init__(
        self,
        uavs_list: List[UAV],
        regions_list: List[Region],
        V_matrix: List[List[float]],
        model_name: str = "gemini-2.5-pro",
        api_key: str = None,
        **kwargs
    ):
        super().__init__(uavs_list, regions_list, V_matrix, **kwargs)
        self.model_name = model_name
        
        # Configure Gemini
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"response_mime_type": "application/json"}
            )
        else:
            self.model = None
            print("Warning: No Google AI Studio API key provided (GEMINI_API_KEY). LLM-APPA will fallback to original APPA.")

    def region_allocation_phase(self) -> Dict[int, List[int]]:
        """
        Phase 1: Region Allocation using LLM (Gemini)
        """
        if not self.model:
            return super().region_allocation_phase()

        # Prepare data for LLM
        uav_data = []
        for i, uav in enumerate(self.uavs_list):
            uav_data.append({
                "id": i,
                "max_velocity": uav.max_velocity,
                "scan_width": uav.scan_width
            })

        region_data = []
        for j, region in enumerate(self.regions_list):
            # Find compatible UAVs (V > 0)
            compatible_uavs = [i for i in range(self.num_uavs) if self.V_matrix[i][j] > 0]
            region_data.append({
                "id": j,
                "coords": region.coords,
                "area": region.area,
                "compatible_uavs": compatible_uavs
            })

        # Construct Prompt
        prompt = f"""
        You are an expert in Multi-UAV Task Allocation.
        Your task is to allocate Regions to UAVs to minimize the Makespan (Max Completion Time).

        **Input Data:**
        **UAVs:** (id, max_velocity, scan_width)
        {json.dumps(uav_data, indent=2)}

        **Regions:** (id, coords, area, compatible_uavs)
        {json.dumps(region_data, indent=2)}

        **Constraints:**
        1. (C1) Each UAV performs at most one mission cycle (Base -> Regions -> Base).
        2. (C2) The number of active UAVs cannot exceed the total available UAVs.
        3. (C3) If a UAV is assigned any regions, it must complete a full tour returning to the base.
        4. (C4) Every region must be visited exactly once by exactly one UAV.
        5. **CRITICAL**: A region `j` can ONLY be assigned to UAV `i` if it is in the `compatible_uavs` list (i.e., Velocity[i][j] > 0).

        **Objective:**
        - Minimize the Makespan (the maximum total time assigned to any single UAV).
        - Scan Time = Region Area / (UAV Velocity * Scan Width).
        - Balance the total Scan Time across UAVs.

        **Output:**
        JSON object: {{"uav_id": [region_ids], ...}}
        Example: {{"0": [1, 2], "1": [3]}}
        """

        try:
            response = self.model.generate_content(prompt)
            content = response.text
            allocation_json = json.loads(content)
            
            # Convert keys to int and ensure all regions are assigned
            region_assignment = {i: [] for i in range(self.num_uavs)}
            assigned_regions = set()

            for uav_id_str, assigned_list in allocation_json.items():
                uav_id = int(uav_id_str)
                if 0 <= uav_id < self.num_uavs:
                    for region_id in assigned_list:
                        if 0 <= region_id < self.num_regions:
                            # Validate compatibility
                            if self.V_matrix[uav_id][region_id] > 0:
                                region_assignment[uav_id].append(region_id)
                                assigned_regions.add(region_id)
                            else:
                                print(f"Warning: LLM assigned Region {region_id} to incompatible UAV {uav_id}. Ignoring.")
            
            # Fallback for unassigned or invalidly assigned regions
            all_regions = set(range(self.num_regions))
            unassigned = all_regions - assigned_regions
            
            if unassigned:
                print(f"Warning: Reassigning {len(unassigned)} unassigned/invalid regions.")
                for region_id in unassigned:
                    # Find compatible UAVs
                    compatible_uavs = [i for i in range(self.num_uavs) if self.V_matrix[i][region_id] > 0]
                    if compatible_uavs:
                        # Assign to random compatible UAV
                        uav_idx = np.random.choice(compatible_uavs)
                        region_assignment[uav_idx].append(region_id)
                    else:
                        print(f"Error: Region {region_id} has no compatible UAVs!")

            return region_assignment

        except Exception as e:
            print(f"Error calling LLM: {e}")
            print("Falling back to original APPA allocation.")
            return super().region_allocation_phase()

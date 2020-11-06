import torch

def convert_instance_mask_to_center_and_regression_label(instance_mask, num_instances, ignore_index=255):
    t, h, w = instance_mask.shape
    instance_ids = range(num_instances)
    center_label = torch.zeros(t, w, h, 1)
    future_displacement_label = torch.zeros(t, w, h, 2)
    regression_label = torch.zeros(t, w, h, 2)

    for instance_id in instance_ids:
        instance_points = torch.nonzero(instance_mask == instance_id)
        prev_center = None
        prev_instance_points_index = None
        for time in range(t):
            instance_points_at_time_index = instance_points[instance_points[:, 0] == time]
            instance_points_at_time_sliced = instance_points_at_time_index[:, 1:].float()
            
            if instance_points_at_time_index.shape[0]:
                max_points = torch.max(instance_points_at_time_sliced, dim=0, keepdims=True)[0]
                min_points = torch.min(instance_points_at_time_sliced, dim=0, keepdims=True)[0]
                instance_extent =  max_points - min_points
                instance_center =  torch.mean(instance_points_at_time_sliced, dim=0, keepdims=True)
                
                instance_points_to_center = instance_points_at_time_sliced - instance_center
                center_heatmap = 1 - torch.mean((2 * torch.abs(instance_points_to_center) / instance_extent), dim=1, keepdims=True)
                regression_label[
                    instance_points_at_time_index[:, 0], 
                    instance_points_at_time_index[:, 1], 
                    instance_points_at_time_index[:, 2]
                ] = instance_points_to_center
                center_label[
                    instance_points_at_time_index[:, 0], 
                    instance_points_at_time_index[:, 1], 
                    instance_points_at_time_index[:, 2]
                ] = center_heatmap
                
                if prev_center is not None:
                    future_displacement_label[
                        prev_instance_points_index[:, 0], 
                        prev_instance_points_index[:, 1], 
                        prev_instance_points_index[:, 2]
                    ] = instance_center - prev_center
                prev_center = instance_center
                prev_instance_points_index = instance_points_at_time_index
            else:
                prev_center = None
                prev_instance_points_index = None 
        
    return center_label, regression_label, future_displacement_label
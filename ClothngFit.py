bl_info = {
    "name": "Clothing Fit Tool",
    "author": "Doxes",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > Clothing Fit Tool",
    "description": "Apply clothing fit to a character",
    "warning": "",
    "wiki_url": "",
    "category": "Object",
}

import bpy
import bmesh
from mathutils import Vector
from math import exp
from mathutils.geometry import distance_point_to_plane
from bpy.props import StringProperty, FloatProperty, EnumProperty
import math


class ClothingFitPanel(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_clothing_fit_tool"
    bl_label = "Clothing Fit Tool"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Clothing Fit Tool"

    def draw(self, context):
        layout = self.layout

        scene = context.scene
        tool_props = scene.clothing_fit_tool_props
        
        layout.prop_search(tool_props, "clothing_obj_name", bpy.data, "objects", text="Clothing Object")
        layout.prop_search(tool_props, "body_obj_name", bpy.data, "objects", text="Body Object")
        layout.prop(tool_props, "vertex_group_name")
        layout.prop(tool_props, "soft_range")
        layout.prop(tool_props, "falloff_type")
        layout.prop(tool_props, "ideal_distance")
        
        layout.prop(tool_props, "projection_type")
        
        if tool_props.projection_type == 'NORMAL' or tool_props.projection_type == 'MIXED':
            layout.prop(tool_props, "ray_offset")

        layout.operator("object.apply_clothing_fit", text="Apply Clothing Fit")
        layout.operator("object.select_intersecting_vertices", text="Select Intersecting")
        
        layout.operator("object.fix_intersecting_vertices", text="Fix Intersecting")
        


class ClothingFitToolProperties(bpy.types.PropertyGroup):
    clothing_obj_name: StringProperty(name="Clothing Object")
    body_obj_name: StringProperty(name="Body Object")
    vertex_group_name: StringProperty(name="Vertex Group")
    soft_range: FloatProperty(name="Soft Range", default=1.0, min=0.0)
    falloff_type: EnumProperty(
        name="Falloff Type",
        items=[
            ('LINEAR', "Linear", ""),
            ('QUADRATIC', "Quadratic", ""),
            ('EXPONENTIAL', "Exponential", ""),
            ('SMOOTHSTEP', "SmoothStep", ""),
            ('COSINE', "Cosine", ""),
        ],
        default='COSINE'
    )
    ideal_distance: FloatProperty(name="Ideal Distance", default=0.1)
    projection_type: EnumProperty(
        name="Projection Type",
        items=[
            ('CENTER', "From Center", "Project rays from center point, this is collapsive"),
            ('NORMAL', "Vertex Normal", "Project rays along vertex normals (auto filtered bad normals)"),
            ('NEAREST', "Nearest Point", "Find the nearest point on body mesh"),
            ('MIXED', "Mixed Method", "Combine multiple projection methods, can be collapsive")
        ],
        default='NORMAL'
    )
    ray_offset: FloatProperty(
        name="Ray Offset",
        description="Distance to offset ray start position along normal direction",
        default=0.1,
        min=0.0
    )


class ApplyClothingFitOperator(bpy.types.Operator):
    bl_idname = "object.apply_clothing_fit"
    bl_label = "Apply Clothing Fit"
    bl_options = {'REGISTER', 'UNDO'}
    
    SELECTED_GROUP_NAME = "AutoSelectedVertices"

    def save_selected_vertices(self, obj):
        if obj.mode != 'EDIT':
            return False
            
        # Get selected vertices in edit mode
        bm = bmesh.from_edit_mesh(obj.data)
        selected_verts = [v.index for v in bm.verts if v.select]
        
        if not selected_verts:
            return False
            
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Create or get vertex group
        if self.SELECTED_GROUP_NAME in obj.vertex_groups:
            selected_group = obj.vertex_groups[self.SELECTED_GROUP_NAME]
            selected_group.remove(range(len(obj.data.vertices)))  # Clear group
        else:
            selected_group = obj.vertex_groups.new(name=self.SELECTED_GROUP_NAME)
        
        # Add selected vertices to group
        for idx in selected_verts:
            selected_group.add([idx], 1.0, 'REPLACE')
            
        return True

    def execute(self, context):
        scene = context.scene
        tool_props = scene.clothing_fit_tool_props

        clothing_obj = bpy.data.objects.get(tool_props.clothing_obj_name)
        body_obj = bpy.data.objects.get(tool_props.body_obj_name)
        
        if not (clothing_obj and body_obj):
            self.report({'ERROR'}, "Invalid object names")
            return {'CANCELLED'}

        # Save selection state if in edit mode
        was_in_edit = False
        if clothing_obj.mode == 'EDIT':
            was_in_edit = True
            has_selection = self.save_selected_vertices(clothing_obj)
            if not has_selection:
                self.report({'ERROR'}, "No vertices selected")
                return {'CANCELLED'}

        # Apply fitting
        apply_clothing_fit(
            tool_props.clothing_obj_name,
            tool_props.body_obj_name,
            self.SELECTED_GROUP_NAME,
            tool_props.soft_range,
            tool_props.falloff_type,
            tool_props.ideal_distance,
            operator=self
        )

        tool_props.vertex_group_name = self.SELECTED_GROUP_NAME

        # We don't do this anymore because some intersecting behavior is desired
        # Apply fix intersecting
        # fix_op = bpy.ops.object.fix_intersecting_vertices
        # fix_op()

        # Return to edit mode if we started there
        if was_in_edit:
            bpy.ops.object.mode_set(mode='EDIT')
            
        bpy.ops.ed.undo_push(message="Apply Clothing Fit")  # 推送撤销操作后的状态

        return {'FINISHED'}
    

class FixIntersectingVerticesOperator(bpy.types.Operator):
    bl_idname = "object.fix_intersecting_vertices"
    bl_label = "Fix Intersecting Vertices"
    bl_options = {'REGISTER', 'UNDO'}
    
    INTERSECTING_GROUP_NAME = "AutoDetectedIntersecting"
    
    def execute(self, context):
        tool_props = context.scene.clothing_fit_tool_props
        clothing_obj = bpy.data.objects.get(tool_props.clothing_obj_name)
        body_obj = bpy.data.objects.get(tool_props.body_obj_name)
    
        if not all([clothing_obj, body_obj]):
            self.report({'ERROR'}, "Invalid objects")
            return {'CANCELLED'}
    
        # 保存当前模式
        original_mode = clothing_obj.mode
    
        # 切换到对象模式
        if original_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # 创建穿模顶点组
        if self.INTERSECTING_GROUP_NAME in clothing_obj.vertex_groups:
            intersecting_group = clothing_obj.vertex_groups[self.INTERSECTING_GROUP_NAME]
            intersecting_group.remove(range(len(clothing_obj.data.vertices)))
        else:
            intersecting_group = clothing_obj.vertex_groups.new(name=self.INTERSECTING_GROUP_NAME)
    
        # 找到穿模顶点
        for i, v in enumerate(clothing_obj.data.vertices):
            if is_vertex_intersecting(v, clothing_obj.matrix_world, body_obj):
                intersecting_group.add([i], 1.0, 'REPLACE')
    
        # 使用 NEAREST 投影方式修复穿模
        old_projection = tool_props.projection_type
        tool_props.projection_type = 'NEAREST'
    
        apply_clothing_fit(
            tool_props.clothing_obj_name,
            tool_props.body_obj_name,
            self.INTERSECTING_GROUP_NAME,
            tool_props.soft_range,
            tool_props.falloff_type,
            tool_props.ideal_distance,
            operator=self
        )
    
        tool_props.projection_type = old_projection


        if original_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode=original_mode)
        return {'FINISHED'}






class SelectIntersectingVerticesOperator(bpy.types.Operator):
    bl_idname = "object.select_intersecting_vertices"
    bl_label = "Select Intersecting Vertices"
    bl_options = {'REGISTER', 'UNDO'}

    def find_connected_intersecting_vertices(self, start_vert, bm, clothing_world_matrix, body_obj, group_center_body, visited):
        # 递归查找相连的穿模顶点
        result = set()
        to_check = {start_vert}
        
        while to_check:
            current = to_check.pop()
            if current in visited:
                continue
                
            visited.add(current)
            
            # 检查当前顶点是否穿模
            if self.is_vertex_intersecting(current, clothing_world_matrix, body_obj, group_center_body):
                result.add(current)
                # 添加相邻顶点到检查列表
                for edge in current.link_edges:
                    other = edge.other_vert(current)
                    if other not in visited:
                        to_check.add(other)
        
        return result

    def is_vertex_intersecting(self, vert, clothing_world_matrix, body_obj, group_center_body):
        # 将顶点转换到身体空间
        vert_world = clothing_world_matrix @ vert.co
        vert_body = body_obj.matrix_world.inverted() @ vert_world
        
        # 计算从中心到顶点的方向
        direction = (vert_body - group_center_body).normalized()
        
        # 射线检测
        _, location, _, index = body_obj.ray_cast(group_center_body, direction)
        
        if index >= 0:
            # 计算射线方向上的距离
            hit_vec = location - group_center_body
            vert_vec = vert_body - group_center_body
            
            # 如果顶点在命中点之后,说明穿模
            # 使用点积判断相对位置
            if hit_vec.length > vert_vec.length and hit_vec.dot(vert_vec) > 0:
                return True
        
        return False

    def find_symmetric_vertices(self, vertices, bm, threshold=0.001):
        # 查找X轴对称的顶点
        symmetric_vertices = set()
        for v in vertices:
            # 在X轴对称位置查找顶点
            mirror_co = Vector((-v.co.x, v.co.y, v.co.z))
            for other_v in bm.verts:
                if other_v not in vertices:  # 避免重复选择
                    if (other_v.co - mirror_co).length < threshold:
                        symmetric_vertices.add(other_v)
        return symmetric_vertices

    def execute(self, context):
        if context.mode != 'EDIT_MESH':
            self.report({'ERROR'}, "Must be in Edit Mode")
            return {'CANCELLED'}
        
        clothing_obj = context.active_object
        body_obj_name = context.scene.clothing_fit_tool_props.body_obj_name
        body_obj = bpy.data.objects.get(body_obj_name)
    
        if not body_obj:
            self.report({'ERROR'}, "Body object not found")
            return {'CANCELLED'}

        # 获取BMesh
        bm = bmesh.from_edit_mesh(clothing_obj.data)
        clothing_world_matrix = clothing_obj.matrix_world
    
        # 检查每个未选中的顶点
        found_any = False
        for v in bm.verts:
            if not v.select and is_vertex_intersecting(v, clothing_world_matrix, body_obj):
                v.select = True
                found_any = True
    
        bmesh.update_edit_mesh(clothing_obj.data)
    
        if not found_any:
            self.report({'INFO'}, "No more intersecting vertices found")
            return {'CANCELLED'}
        
        return {'FINISHED'}



def get_bbox_center(vertices):
    """计算顶点列表的包围盒中心"""
    if not vertices:
        return Vector((0, 0, 0))
    
    # 获取所有顶点的坐标
    coords = [v.co for v in vertices]
    
    # 计算包围盒的最小和最大点
    min_x = min(co.x for co in coords)
    min_y = min(co.y for co in coords)
    min_z = min(co.z for co in coords)
    max_x = max(co.x for co in coords)
    max_y = max(co.y for co in coords)
    max_z = max(co.z for co in coords)
    
    # 返回包围盒中心
    return Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2))




def is_vertex_intersecting(vert, clothing_world_matrix, body_obj):
    # 转换到body空间
    vert_world = clothing_world_matrix @ vert.co
    vert_body = body_obj.matrix_world.inverted() @ vert_world
    
    # 获取最近点
    success, location, normal, _ = body_obj.closest_point_on_mesh(vert_body)
    if not success:
        return False
        
    # 从最近点到顶点的向量
    to_vert = vert_body - location
    
    # 如果点在表面背面（与法线夹角>90度），就是穿模
    return to_vert.dot(normal) < 0




## not used
def get_interpolated_normal(body_obj, position):
    """
    获取位置处的插值法线
    使用最近的顶点法线进行插值
    position应该在局部坐标系中
    """
    # 转换position到局部坐标
    local_pos = body_obj.matrix_world.inverted() @ position
    
    # 获取最近的面
    success, location, normal, face_idx = body_obj.closest_point_on_mesh(local_pos)
    if not success:
        return None
        
    face = body_obj.data.polygons[face_idx]
    
    # 获取重心坐标
    face_verts = [body_obj.data.vertices[i] for i in face.vertices]
    weights = barycentric_weights(local_pos, 
                                face_verts[0].co,
                                face_verts[1].co,
                                face_verts[2].co)
    
    # 插值法线
    normal = Vector((0, 0, 0))
    for vert, weight in zip(face_verts, weights):
        normal += vert.normal * weight
    
    # 转换法线到世界空间
    world_normal = body_obj.matrix_world.to_3x3() @ normal
    
    return world_normal.normalized()


def barycentric_weights(p, a, b, c):
    """
    计算点p相对于三角形abc的重心坐标
    所有点都应该在同一坐标系中
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-6:
        return (1.0/3.0, 1.0/3.0, 1.0/3.0)  # 退化情况
        
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    return (u, v, w)
## not used





#This keeps only the "good" normal projects, most of them should be reliable
def validate_normal_direction(body_obj, clothing_world_matrix, vert, body_normal, 
                            neighbor_max_angle=math.radians(15), 
                            body_max_angle=math.radians(75), 
                            max_distance=0.03):
    """
    验证normal方向的可靠性
    """
    # 获取world space位置和方向
    vert_world = clothing_world_matrix @ vert.co
    normal_world = clothing_world_matrix.to_3x3() @ vert.normal
    normal_world.normalize()
    
    # 检查与body normal的夹角 - 更宽松的阈值
    body_normal_world = body_obj.matrix_world.to_3x3() @ body_normal
    normal_body_angle = normal_world.angle(body_normal_world)
    if normal_body_angle > body_max_angle:
        return False
    
    # 检查与邻居normal的夹角
    # 获取所有邻居
    neighbors = [e.other_vert(vert) for e in vert.link_edges]
    neighbor_count = len(neighbors)
        
    # 计算与每个邻居normal的夹角
    bad_neighbor_count = 0
    for n in neighbors:
        n_normal_world = clothing_world_matrix.to_3x3() @ n.normal
        n_normal_world.normalize()
        angle = normal_world.angle(n_normal_world)
        
        if angle > neighbor_max_angle:
            bad_neighbor_count += 1
            # 3个邻居时不允许任何bad neighbor
            if neighbor_count <= 3 or bad_neighbor_count > 1:
                return False
    
    # 检查world space的raycast距离
    ray_origin = vert_world + normal_world * 0.1
    ray_dir = -normal_world
    success, hit_pos, _, _ = body_obj.ray_cast(body_obj.matrix_world.inverted() @ ray_origin, 
                                              body_obj.matrix_world.inverted().to_3x3() @ ray_dir)
    
    if success:
        hit_world = body_obj.matrix_world @ hit_pos
        distance = (hit_world - vert_world).length
        if distance > max_distance:
            return False
    else:
        return False
        
    return True





def get_bone_line_segments(vert, clothing_obj, operator):
    """
    获取所有相关骨骼的线段信息
    返回: [(start_pos, end_pos, weight, bone_name), ...]
    """
    bone_weights = {}
    armature = clothing_obj.parent
    if not armature:
        operator.report({'INFO'}, "No armature found")
        return []

    # 收集所有权重大于阈值的骨骼
    significant_bones = {}
    for vg in clothing_obj.vertex_groups:
        if vg.name in armature.pose.bones:
            try:
                weight = vg.weight(vert.index)
                if weight > 0.01:  # 显著权重阈值
                    significant_bones[vg.name] = weight
            except RuntimeError:
                continue
    
    if not significant_bones:
        return []
    
    
    operator.report({'INFO'}, f"Vertex {vert.index} has bones: {significant_bones}")
        
    # 如果只有一个骨骼，直接返回其位置
    if len(significant_bones) == 1:
        bone_name = next(iter(significant_bones))
        bone = armature.pose.bones[bone_name]
        pos = armature.matrix_world @ bone.head
        return [(pos, pos, significant_bones[bone_name], bone_name)]  # 单点情况

    # 构建骨骼线段
    line_segments = []
    for bone_name, weight in significant_bones.items():
        bone = armature.pose.bones[bone_name]
        parent = bone.parent
        
        # 如果父骨骼也有权重，创建完整线段
        if parent and parent.name in significant_bones:
            start = armature.matrix_world @ parent.head
            end = armature.matrix_world @ bone.head
            # 使用两个骨骼的平均权重
            avg_weight = (weight + significant_bones[parent.name]) / 2
            line_segments.append((start, end, avg_weight, f"{parent.name}-{bone_name}"))
    
    operator.report({'INFO'}, f"Found {len(line_segments)} valid segments")
    return line_segments

def project_point_to_line(point, line_start, line_end):
    """
    计算点到线段的垂直投影
    返回: (投影点, 是否在线段上, 到线段距离)
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_length = line_vec.length
    if line_length == 0:
        return line_start, True, point_vec.length
        
    # 计算投影
    t = line_vec.dot(point_vec) / (line_length * line_length)
    
    # 检查是否在线段上
    if t < 0:
        proj_point = line_start
        on_segment = False
    elif t > 1:
        proj_point = line_end
        on_segment = False
    else:
        proj_point = line_start + line_vec * t
        on_segment = True
        
    distance = (point - proj_point).length
    return proj_point, on_segment, distance

def calculate_projection_points(vert_world_pos, bone_segments):
    """
    计算所有投影点及其权重
    返回: [(投影点, 权重), ...]
    """
    projection_results = []
    
    for start, end, bone_weight, bone_name in bone_segments:
        proj_point, on_segment, distance = project_point_to_line(vert_world_pos, start, end)
        
        # 计算混合权重
        # 使用距离的倒数和骨骼权重的组合
        dist_factor = 1.0 / (1.0 + distance)  # 避免除以零
        combined_weight = bone_weight * dist_factor
        
        projection_results.append((proj_point, combined_weight, bone_name))
        
    return projection_results

def get_final_projection_start(projection_points):
    """
    计算最终的投射起点
    """
    if not projection_points:
        return None
        
    total_weight = sum(weight for _, weight, _ in projection_points)
    if total_weight == 0:
        return None
        
    weighted_pos = Vector((0, 0, 0))
    for pos, weight, _ in projection_points:
        weighted_pos += pos * (weight / total_weight)
        
    return weighted_pos



def get_bone_chain_and_distances(vert, clothing_obj, operator):
    """
    获取骨骼链和顶点到每个骨骼的距离
    返回: (bone_positions, distances)
    bone_positions: 骨骼链上所有点的世界空间位置列表
    distances: 顶点到每个骨骼点的距离列表
    """
    armature = clothing_obj.parent
    if not armature:
        operator.report({'INFO'}, "No armature found")
        return None, None

    # 找到最大权重的骨骼
    max_weight_bone = None
    max_weight = 0
    for vg in clothing_obj.vertex_groups:
        if vg.name in armature.pose.bones:
            try:
                weight = vg.weight(vert.index)
                if weight > max_weight:
                    max_weight = weight
                    max_weight_bone = armature.pose.bones[vg.name]
            except RuntimeError:
                continue
    
    if not max_weight_bone:
        return None, None

    # 构建骨骼链（向上和向下各找一个骨骼）
    bone_chain = []
    if max_weight_bone.parent:
        bone_chain.append(max_weight_bone.parent)
    bone_chain.append(max_weight_bone)
    if max_weight_bone.children:
        bone_chain.append(max_weight_bone.children[0])
    
    # 获取骨骼位置和距离
    vert_world_pos = clothing_obj.matrix_world @ vert.co
    bone_positions = []
    distances = []
    
    for bone in bone_chain:
        bone_pos = armature.matrix_world @ bone.head
        bone_positions.append(bone_pos)
        distances.append((vert_world_pos - bone_pos).length)
    
    return bone_positions, distances

def catmull_rom_point(t, p0, p1, p2, p3):
    """
    计算Catmull-Rom样条上的点
    """
    t2 = t * t
    t3 = t2 * t
    
    # Catmull-Rom矩阵乘法展开
    m = [-0.5, 1.5, -1.5, 0.5,
         1.0, -2.5, 2.0, -0.5,
         -0.5, 0.0, 0.5, 0.0,
         0.0, 1.0, 0.0, 0.0]
    
    q = Vector((0, 0, 0))
    q += (m[0] * t3 + m[4] * t2 + m[8] * t + m[12]) * p0
    q += (m[1] * t3 + m[5] * t2 + m[9] * t + m[13]) * p1
    q += (m[2] * t3 + m[6] * t2 + m[10] * t + m[14]) * p2
    q += (m[3] * t3 + m[7] * t2 + m[11] * t + m[15]) * p3
    
    return q










def apply_clothing_fit(clothing_obj_name, body_obj_name, vertex_group_name, soft_range, falloff_type='LINEAR', ideal_distance=0.1, center_override=None, operator=None):
    # 获取衣服和身体的BMesh
    clothing_obj = bpy.data.objects[clothing_obj_name]
    body_obj = bpy.data.objects[body_obj_name]
    clothing_bm = bmesh.new()
    clothing_bm.from_mesh(clothing_obj.data)
    clothing_bm.verts.ensure_lookup_table()
    
    clothing_world_matrix = clothing_obj.matrix_world
    clothing_world_matrix_inv = clothing_world_matrix.inverted()    #inv @ worldPos = local
    body_world_matrix = body_obj.matrix_world
    body_world_matrix_inv = body_world_matrix.inverted()
    
    # 找到顶点组
    group_index = clothing_obj.vertex_groups[vertex_group_name].index
    deform_layer = clothing_bm.verts.layers.deform.verify()
    group_verts = [v for v in clothing_bm.verts if group_index in v[deform_layer].keys()]
    
    projection_type = bpy.context.scene.clothing_fit_tool_props.projection_type
    ray_offset = bpy.context.scene.clothing_fit_tool_props.ray_offset
    
    if center_override is not None:
        group_center_body = center_override
    else:
        group_center_local = get_bbox_center(group_verts)
        group_center_world = clothing_world_matrix @ group_center_local
        #group_center_world = bpy.context.scene.cursor.location
        group_center_body = body_world_matrix_inv @ group_center_world
    
    # 存储顶点组外顶点相对于每个顶点组内顶点的权重
    vert_weights = {v: {} for v in clothing_bm.verts } # if v not in group_verts
    
    # 计算权重
    for v in vert_weights:
        total_weight = 0
        max_weight = 0
        for gv in group_verts:
            dist = (v.co - gv.co).length
            if dist < soft_range:
                x = dist / soft_range
                if falloff_type == 'LINEAR':
                    weight = 1 - x
                elif falloff_type == 'QUADRATIC':
                    weight = 1 - (x) ** 2
                elif falloff_type == 'EXPONENTIAL':
                    weight = exp(-x)
                elif falloff_type == 'SMOOTHSTEP':
                    weight = 1 - (3*x*x - 2*x*x*x)
                elif falloff_type == 'COSINE':
                    weight = 0.5 * (1 - math.cos(math.pi * (1 - x)))
                else:
                    raise ValueError(f"Unsupported falloff type: {falloff_type}")
                vert_weights[v][gv] = weight
                max_weight = max(weight, max_weight)
                total_weight += weight
        
        # 皈依化权重
        if total_weight > max_weight:
            scale = max_weight / total_weight
            for gv in vert_weights[v]:
                vert_weights[v][gv] *= scale
                 
                 
    clothing_obj = bpy.data.objects[clothing_obj_name]
    
    # 移动顶点组内的顶点并带动其他顶点
    for gv in group_verts:
        
        gv_co_world = clothing_world_matrix @ gv.co
        gv_co_body = body_world_matrix_inv @ gv_co_world
        
        
        if projection_type == 'CENTER':
        
            # 获取骨骼链和距离
            bone_positions, distances = get_bone_chain_and_distances(gv, clothing_obj, operator)
        
            if not bone_positions or len(bone_positions) < 2:
                # 如果没有足够的骨骼，使用最近点
                gv_co_body = body_world_matrix_inv @ gv_co_world
                _, location, normal, _ = body_obj.closest_point_on_mesh(gv_co_body)
                ideal_pos_body = location + normal * ideal_distance
                ideal_pos_world = body_world_matrix @ ideal_pos_body
                ideal_pos = clothing_world_matrix_inv @ ideal_pos_world
                delta = ideal_pos - gv.co
            
                for v, gv_weights in vert_weights.items():
                    if gv in gv_weights:
                        v.co += delta * gv_weights[gv]
                continue
            
            # 计算参数t：使用到最近两个骨骼的距离比例
            min_dist_idx = distances.index(min(distances))
            if min_dist_idx == len(distances) - 1:
                min_dist_idx = len(distances) - 2
        
            d1 = distances[min_dist_idx]
            d2 = distances[min_dist_idx + 1]
            t = d1 / (d1 + d2)
        
            # 确保有足够的点来构建Catmull-Rom样条
            if len(bone_positions) < 4:
                if len(bone_positions) == 2:
                    p0 = bone_positions[0] + (bone_positions[0] - bone_positions[1])
                    p3 = bone_positions[1] + (bone_positions[1] - bone_positions[0])
                else:  # 3个点
                    p0 = bone_positions[0] + (bone_positions[0] - bone_positions[1])
                    p3 = bone_positions[2]
            else:
                p0 = bone_positions[0]
                p3 = bone_positions[3]
        
            # 计算样条上的投影起点
            ray_start_world = catmull_rom_point(t, p0, bone_positions[min_dist_idx],
                                          bone_positions[min_dist_idx + 1], p3)
        
            # 转换到body空间并进行投射
            ray_start_body = body_world_matrix_inv @ ray_start_world
            gv_co_body = body_world_matrix_inv @ gv_co_world
            direction = (gv_co_body - ray_start_body).normalized()
        
            # 射线检测和顶点移动
            _, location, normal, index = body_obj.ray_cast(ray_start_body, direction)
            if index < 0:
                continue
            ideal_pos_body = location + normal * ideal_distance
            ideal_pos_world = body_world_matrix @ ideal_pos_body
            if (ideal_pos_world - gv_co_world).length > 0.1:
                continue
            
        elif projection_type == 'NORMAL' or projection_type == 'MIXED':
            normal_local = gv.normal
            normal_world = clothing_world_matrix.to_3x3() @ normal_local
            normal_body = body_world_matrix_inv.to_3x3() @ normal_world
            direction = -normal_body.normalized()  # 反方向
            # 从顶点位置向外偏移ray_offset距离作为射线起点
            ray_start = gv_co_body - direction * ray_offset
            
            _, location, normal, index = body_obj.ray_cast(ray_start, direction)
            # 选择有效的射线结果
            if index < 0:
                continue
            
            if not validate_normal_direction(body_obj, clothing_world_matrix, gv, normal):
                if projection_type == 'NORMAL':
                    continue
                else:
                    # 使用最近点
                    _, location, normal, _ = body_obj.closest_point_on_mesh(gv_co_body)
        
        
        elif projection_type == 'NEAREST':
            success, location, normal, _ = body_obj.closest_point_on_mesh(gv_co_body)
            if not success:
                continue
        
        ideal_pos_body = location + normal * ideal_distance
        ideal_pos_world = body_world_matrix @ ideal_pos_body
        ideal_pos = clothing_world_matrix_inv @ ideal_pos_world
        
        # 移动顶点组内的顶点
        delta = ideal_pos - gv.co
        
        # 带动顶点组外的顶点
        for v, gv_weights in vert_weights.items():
            if gv in gv_weights:
                v.co += delta * gv_weights[gv]

    # 更新网格            
    clothing_bm.to_mesh(clothing_obj.data)
    clothing_bm.free()










def register():
    bpy.utils.register_class(ClothingFitPanel)
    bpy.utils.register_class(ClothingFitToolProperties)
    bpy.utils.register_class(SelectIntersectingVerticesOperator)
    bpy.types.Scene.clothing_fit_tool_props = bpy.props.PointerProperty(type=ClothingFitToolProperties)
    bpy.utils.register_class(ApplyClothingFitOperator)
    bpy.utils.register_class(FixIntersectingVerticesOperator)

def unregister():
    bpy.utils.unregister_class(ClothingFitPanel)
    bpy.utils.unregister_class(ClothingFitToolProperties)
    bpy.utils.unregister_class(SelectIntersectingVerticesOperator)
    del bpy.types.Scene.clothing_fit_tool_props
    bpy.utils.unregister_class(ApplyClothingFitOperator)
    bpy.utils.unregister_class(AutoBoneFitOperator)
    bpy.utils.unregister_class(FixIntersectingVerticesOperator)

if __name__ == "__main__":
    register()

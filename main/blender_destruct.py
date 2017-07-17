import bpy
import json


def export_all_obj(exportFolder):
    meta = {}
    objects = bpy.data.objects
    for object in objects:
        meta.update({object.name: {'loc': tuple(object.location),
                                   'dim': tuple(object.dimensions)}})
        bpy.ops.object.select_all(action='DESELECT')
        object.select = True
        exportName = exportFolder + '\\' + object.name + '.obj'
        bpy.ops.export_scene.obj(filepath=exportName, use_selection=True)
    with open(exportFolder + '\\meta.json', 'w') as f:
        json.dump(meta, f)


export_all_obj(r'C:\blender_new_output')

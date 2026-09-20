# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full-mesh EGL renderer using installed system libraries and NumPy only."""

from __future__ import annotations

import ctypes as C
import os

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normals(vertices, faces):
    triangle = vertices[faces]
    normals = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    result = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(result, faces[:, corner], normals)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return result


class Renderer:
    """Render every source triangle with a shared depth buffer and smooth lighting."""

    def __init__(self, device="cpu", width=960, height=464):
        import torch

        self.device = torch.device("cpu")
        self.width, self.height = width, height
        self.egl = C.CDLL("libEGL.so.1")
        self.gl = C.CDLL("libGLESv2.so.2")
        self._bind()
        address = self.egl.eglGetProcAddress(b"eglGetPlatformDisplayEXT")
        get_display = C.CFUNCTYPE(C.c_void_p, C.c_uint, C.c_void_p, C.POINTER(C.c_int))(address)
        platform = os.environ.get("MUG_EGL_PLATFORM", "surfaceless")
        self.display = get_display({"surfaceless": 0x31DD, "x11": 0x31D5}[platform], None, None)
        major, minor = C.c_int(), C.c_int()
        if not self.egl.eglInitialize(self.display, C.byref(major), C.byref(minor)):
            raise RuntimeError("EGL initialization failed")
        config_attributes = (C.c_int * 17)(
            0x3033,
            1,
            0x3040,
            0x40,
            0x3024,
            8,
            0x3023,
            8,
            0x3022,
            8,
            0x3021,
            8,
            0x3025,
            24,
            0x3026,
            0,
            0x3038,
        )
        config, count = C.c_void_p(), C.c_int()
        if (
            not self.egl.eglChooseConfig(self.display, config_attributes, C.byref(config), 1, C.byref(count))
            or not count.value
        ):
            raise RuntimeError("EGL has no RGB/depth GLES3 configuration")
        self.egl.eglBindAPI(0x30A0)
        context_attributes = (C.c_int * 3)(0x3098, 3, 0x3038)
        self.context = self.egl.eglCreateContext(self.display, config, None, context_attributes)
        surface_attributes = (C.c_int * 5)(0x3057, width, 0x3056, height, 0x3038)
        self.surface = self.egl.eglCreatePbufferSurface(self.display, config, surface_attributes)
        if (
            not self.context
            or not self.surface
            or not self.egl.eglMakeCurrent(self.display, self.surface, self.surface, self.context)
        ):
            raise RuntimeError("EGL context creation failed")
        self.backend = self.gl.glGetString(0x1F01).decode()
        self.renderer_info = {"api": "OpenGL ES 3 / EGL", "renderer": self.backend, "native_meshes": True}
        self.program = self._program()
        self.gl.glUseProgram(self.program)
        self.locations = {
            name: self.gl.glGetUniformLocation(self.program, name.encode())
            for name in ("model", "vp", "base_color", "light_direction")
        }
        self.gl.glEnable(0x0B71)  # DEPTH_TEST
        self.gl.glDepthFunc(0x0201)  # LESS
        self.gl.glClearColor(0.973, 0.980, 0.988, 1.0)
        self.gl.glViewport(0, 0, width, height)
        self.gl.glPixelStorei(0x0D05, 1)
        self._buffers = {}
        self.gl.glEnableVertexAttribArray(0)
        self.gl.glEnableVertexAttribArray(1)
        self._cache_identity = None
        self._cached = None

    def _bind(self):
        void, integer, uint, float_ = C.c_void_p, C.c_int, C.c_uint, C.c_float
        ip, up = C.POINTER(integer), C.POINTER(uint)
        bindings = {
            "eglGetProcAddress": (void, [C.c_char_p]),
            "eglInitialize": (uint, [void, ip, ip]),
            "eglChooseConfig": (uint, [void, ip, C.POINTER(void), integer, ip]),
            "eglBindAPI": (uint, [uint]),
            "eglCreateContext": (void, [void, void, void, ip]),
            "eglCreatePbufferSurface": (void, [void, void, ip]),
            "eglMakeCurrent": (uint, [void, void, void, void]),
        }
        for name, (result, args) in bindings.items():
            function = getattr(self.egl, name)
            function.restype, function.argtypes = result, args
        bindings = {
            "glGetString": (C.c_char_p, [uint]),
            "glCreateShader": (uint, [uint]),
            "glShaderSource": (None, [uint, integer, C.POINTER(C.c_char_p), ip]),
            "glCompileShader": (None, [uint]),
            "glGetShaderiv": (None, [uint, uint, ip]),
            "glGetShaderInfoLog": (None, [uint, integer, ip, void]),
            "glCreateProgram": (uint, []),
            "glAttachShader": (None, [uint, uint]),
            "glLinkProgram": (None, [uint]),
            "glGetProgramiv": (None, [uint, uint, ip]),
            "glUseProgram": (None, [uint]),
            "glGetUniformLocation": (integer, [uint, C.c_char_p]),
            "glUniformMatrix4fv": (None, [integer, integer, C.c_ubyte, void]),
            "glUniform3f": (None, [integer, float_, float_, float_]),
            "glEnable": (None, [uint]),
            "glDepthFunc": (None, [uint]),
            "glClearColor": (None, [float_, float_, float_, float_]),
            "glClear": (None, [uint]),
            "glViewport": (None, [integer, integer, integer, integer]),
            "glPixelStorei": (None, [uint, integer]),
            "glGenBuffers": (None, [integer, up]),
            "glBindBuffer": (None, [uint, uint]),
            "glBufferData": (None, [uint, C.c_ssize_t, void, uint]),
            "glEnableVertexAttribArray": (None, [uint]),
            "glVertexAttribPointer": (None, [uint, integer, uint, C.c_ubyte, integer, void]),
            "glDrawElements": (None, [uint, integer, uint, void]),
            "glReadPixels": (None, [integer, integer, integer, integer, uint, uint, void]),
            "glGetError": (uint, []),
        }
        for name, (result, args) in bindings.items():
            function = getattr(self.gl, name)
            function.restype, function.argtypes = result, args

    def _program(self):
        vertex = b"""#version 300 es
        layout(location=0) in vec3 position;
        layout(location=1) in vec3 normal;
        uniform mat4 model; uniform mat4 vp;
        out vec3 world_normal;
        void main(){ gl_Position=vp*model*vec4(position,1.0);
                     world_normal=mat3(model)*normal; }
        """
        fragment = b"""#version 300 es
        precision highp float;
        in vec3 world_normal;
        uniform vec3 base_color; uniform vec3 light_direction;
        out vec4 color;
        void main(){float diffuse=abs(dot(normalize(world_normal),normalize(light_direction)));
                    color=vec4(base_color*(0.58+0.42*diffuse)+0.04*pow(diffuse,24.0),1.0);}
        """
        program = self.gl.glCreateProgram()
        for stage, source in ((0x8B31, vertex), (0x8B30, fragment)):
            shader = self.gl.glCreateShader(stage)
            pointer = C.c_char_p(source)
            self.gl.glShaderSource(shader, 1, C.byref(pointer), None)
            self.gl.glCompileShader(shader)
            success = C.c_int()
            self.gl.glGetShaderiv(shader, 0x8B81, C.byref(success))
            if not success.value:
                log = C.create_string_buffer(4096)
                self.gl.glGetShaderInfoLog(shader, 4096, None, log)
                raise RuntimeError(log.value.decode())
            self.gl.glAttachShader(program, shader)
        self.gl.glLinkProgram(program)
        success = C.c_int()
        self.gl.glGetProgramiv(program, 0x8B82, C.byref(success))
        if not success.value:
            raise RuntimeError("GLES shader linking failed")
        return program

    @staticmethod
    def _view_projection(spec):
        eye, target, up = map(np.asarray, (spec["eye"], spec["target"], spec["up"]))
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        view = np.eye(4)
        view[:3, :3] = np.stack([right, up, -forward])
        view[:3, 3] = -view[:3, :3] @ eye
        near, far = spec["znear"], spec["zfar"]
        projection = np.diag([2 / spec["width"], 2 / spec["height"], -2 / (far - near), 1.0])
        projection[2, 3] = -(far + near) / (far - near)
        return projection @ view

    def _matrix(self, key, value):
        column_major = np.ascontiguousarray(np.asarray(value, dtype=np.float32).T)
        self.gl.glUniformMatrix4fv(self.locations[key], 1, 0, column_major.ctypes.data)

    def _upload_mesh(self, name, vertices, normals, faces=None):
        if name not in self._buffers:
            buffers = (C.c_uint * 2)()
            self.gl.glGenBuffers(2, buffers)
            self._buffers[name] = [buffers[0], buffers[1], 0]
        buffers = self._buffers[name]
        attributes = np.ascontiguousarray(np.concatenate([vertices, normals], axis=1), dtype=np.float32)
        self.gl.glBindBuffer(0x8892, buffers[0])
        self.gl.glBufferData(0x8892, attributes.nbytes, attributes.ctypes.data, 0x88E0)
        if faces is not None:
            indices = np.ascontiguousarray(faces, dtype=np.uint32)
            self.gl.glBindBuffer(0x8893, buffers[1])
            self.gl.glBufferData(0x8893, indices.nbytes, indices.ctypes.data, 0x88E4)
            buffers[2] = indices.size

    def _draw(self, name, color, model):
        vbo, ebo, count = self._buffers[name]
        self.gl.glBindBuffer(0x8892, vbo)
        self.gl.glVertexAttribPointer(0, 3, 0x1406, 0, 24, None)
        self.gl.glVertexAttribPointer(1, 3, 0x1406, 0, 24, C.c_void_p(12))
        self.gl.glBindBuffer(0x8893, ebo)
        self._matrix("model", model)
        self.gl.glUniform3f(self.locations["base_color"], *color)
        self.gl.glDrawElements(0x0004, count, 0x1405, None)

    def render_clip(self, data, specs, start, end):
        """Return RGB panels with shape (frames, 4, height, width, 3)."""
        if data is not self._cache_identity:
            arrays = {key: _array(value) for key, value in data.items()}
            object_count = len(arrays["object_vertices_canonical"])
            if "object_faces" not in arrays:
                scene = arrays["original_scene_faces"]
                arrays["object_faces"] = scene[np.all(scene < object_count, axis=1)]
                for side in ("original", "soma"):
                    scene = arrays[f"{side}_scene_faces"]
                    arrays[f"{side}_faces"] = scene[np.all(scene >= object_count, axis=1)] - object_count
            arrays["object_normals"] = _normals(arrays["object_vertices_canonical"], arrays["object_faces"])
            self._upload_mesh(
                "object", arrays["object_vertices_canonical"], arrays["object_normals"], arrays["object_faces"]
            )
            for side in ("original", "soma"):
                vertices = arrays[f"{side}_vertices"][0]
                self._upload_mesh(side, vertices, np.zeros_like(vertices), arrays[f"{side}_faces"])
            self._cached, self._cache_identity = arrays, data
        arrays = self._cached
        upright = np.asarray(specs["canonical_to_z_up"])
        result = np.empty((end - start, 4, self.height, self.width, 3), dtype=np.uint8)
        pixels = np.empty((self.height, self.width, 4), dtype=np.uint8)
        projections = {name: self._view_projection(specs[name]) for name in ("overview", "detail")}
        world_detail_camera = specs["detail"]["coordinate_frame"] == "world_follow_handle"
        colors = {"original": (0.22, 0.52, 0.88), "soma": (0.96, 0.53, 0.16)}
        for output_index, frame in enumerate(range(start, end)):
            rotation, translation = arrays["object_rotation"][frame], arrays["object_translation"][frame]
            object_world = np.eye(4)
            object_world[:3, :3], object_world[:3, 3] = rotation, translation
            object_detail = np.eye(4)
            object_detail[:3, :3] = upright
            world_detail = np.eye(4)
            world_detail[:3, :3] = upright @ rotation.T
            world_detail[:3, 3] = -upright @ rotation.T @ translation
            normals = {
                side: _normals(arrays[f"{side}_vertices"][frame], arrays[f"{side}_faces"])
                for side in ("original", "soma")
            }
            for side in ("original", "soma"):
                self._upload_mesh(side, arrays[f"{side}_vertices"][frame], normals[side])
            for row, view in enumerate(("overview", "detail")):
                frame_spec = specs["detail_frames"][frame] if view == "detail" and world_detail_camera else specs[view]
                projection = (
                    self._view_projection(frame_spec) if view == "detail" and world_detail_camera else projections[view]
                )
                self._matrix("vp", projection)
                self.gl.glUniform3f(self.locations["light_direction"], *frame_spec["view_direction"])
                use_world = view == "overview" or world_detail_camera
                for column, side in enumerate(("original", "soma")):
                    self.gl.glClear(0x00004000 | 0x00000100)
                    self._draw(
                        "object",
                        (0.52, 0.63, 0.60),
                        object_world if use_world else object_detail,
                    )
                    self._draw(
                        side,
                        colors[side],
                        np.eye(4) if use_world else world_detail,
                    )
                    self.gl.glReadPixels(0, 0, self.width, self.height, 0x1908, 0x1401, pixels.ctypes.data)
                    result[output_index, row * 2 + column] = pixels[::-1, :, :3]
        error = self.gl.glGetError()
        if error:
            raise RuntimeError(f"OpenGL rendering failed with error 0x{error:x}")
        return result


GLRenderer = Renderer

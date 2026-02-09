use core::f32;
use core::f32::consts::FRAC_PI_2;

use bevy::{
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    reflect::TypePath,
    render::{
        mesh::{MeshVertexBufferLayoutRef, PrimitiveTopology},
        primitives::Aabb,
        render_asset::RenderAssetUsages,
        render_resource::{
            AsBindGroup, PolygonMode, RenderPipelineDescriptor, ShaderRef,
            SpecializedMeshPipelineError,
        },
        view::NoFrustumCulling,
    },
    window::PrimaryWindow,
};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin, TouchControls};
use once_cell::sync::Lazy;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use uuid::Uuid;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Firefox / Gecko wheel-event normalisation
// ---------------------------------------------------------------------------
#[wasm_bindgen(inline_js = "
export function install_gecko_wheel_fix(canvas_selector) {
    const PIXELS_PER_LINE = 20;
    const canvas = document.querySelector(canvas_selector);
    if (!canvas) return;

    canvas.addEventListener('wheel', function(e) {
        if (e.deltaMode === 1) {
            Object.defineProperty(e, 'deltaMode', { value: 0 });
            Object.defineProperty(e, 'deltaX',    { value: e.deltaX * PIXELS_PER_LINE });
            Object.defineProperty(e, 'deltaY',    { value: e.deltaY * PIXELS_PER_LINE });
            Object.defineProperty(e, 'deltaZ',    { value: e.deltaZ * PIXELS_PER_LINE });
        }
    }, { capture: true });
}
")]
extern "C" {
    fn install_gecko_wheel_fix(canvas_selector: &str);
}

// ---------------------------------------------------------------------------
// Performance constants
// ---------------------------------------------------------------------------
const AXIS_LENGTH: f32 = 500.;
const VOLUME_CENTER: Vec3 = Vec3::new(0.5, 0.5, 0.5);

/// Distance at which points start to fade
const FADE_START_DISTANCE: f32 = 2.0;
/// Distance at which points are fully faded (alpha = MIN_ALPHA)
const FADE_END_DISTANCE: f32 = 5.0;
/// Minimum alpha for distant points (0 = hidden, 0.1 = barely visible)
const MIN_ALPHA: f32 = 0.1;
/// Distance beyond which points are hidden entirely (saves GPU)
const CULL_DISTANCE: f32 = 8.0;

/// LOD thresholds - switch to simpler meshes at these distances
const LOD_HIGH_DISTANCE: f32 = 1.5; // Use high-detail mesh
const LOD_MED_DISTANCE: f32 = 3.0; // Use medium-detail mesh
                                   // Beyond LOD_MED_DISTANCE: Use low-detail mesh

/// Sphere detail levels (segments)
const SPHERE_SEGMENTS_HIGH: u32 = 16;
const SPHERE_SEGMENTS_MED: u32 = 8;
const SPHERE_SEGMENTS_LOW: u32 = 4;

// ---------------------------------------------------------------------------
// Static queues for JS interop
// ---------------------------------------------------------------------------
static EMBEDDING_QUEUE: Lazy<Mutex<Vec<JsEmbedding>>> = Lazy::new(|| Mutex::new(Vec::new()));
static DELETE_QUEUE: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(Vec::new()));
static EDGE_QUEUE: Lazy<Mutex<Vec<JsEdge>>> = Lazy::new(|| Mutex::new(Vec::new()));
static EDGE_DELETE_QUEUE: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(Vec::new()));
static RENDERED_EMBEDDINGS: Lazy<Mutex<Vec<TextEmbedding>>> = Lazy::new(|| Mutex::new(Vec::new()));
static RENDERED_EDGES: Lazy<Mutex<Vec<RenderedEdge>>> = Lazy::new(|| Mutex::new(Vec::new()));
static SELECTED_EMBEDDING_ID: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

pub const LINE_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(0xA3E0_9C7D_8B51_42C3_9F77_12AB_34CD_5678);

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsEmbedding {
    pub id: String,
    pub position: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsEdge {
    pub id: String,
    pub from: [f32; 3],
    pub to: [f32; 3],
    pub color: Option<[f32; 4]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RenderedEdge {
    id: String,
    from: [f32; 3],
    to: [f32; 3],
    drawn: bool,
}

#[derive(Component, Debug, Clone, Serialize, Deserialize)]
struct TextEmbedding {
    id: String,
    embedding: [f32; 3],
    drawn: bool,
}

#[derive(Component)]
struct Selected;

#[derive(Component)]
struct EdgeEntity {
    #[allow(dead_code)]
    id: String,
}

/// Track current LOD level to avoid unnecessary mesh swaps
#[derive(Component, Default, PartialEq, Eq, Clone, Copy)]
enum LodLevel {
    #[default]
    High,
    Medium,
    Low,
}

/// Shared mesh handles for LOD levels
#[derive(Resource)]
struct SharedMeshes {
    sphere_high: Handle<Mesh>,
    sphere_med: Handle<Mesh>,
    sphere_low: Handle<Mesh>,
}

#[derive(Asset, TypePath, Default, AsBindGroup, Debug, Clone)]
struct LineMaterial {
    #[uniform(0)]
    color: LinearRgba,
}

impl Material for LineMaterial {
    fn fragment_shader() -> ShaderRef {
        LINE_SHADER_HANDLE.clone().into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.polygon_mode = PolygonMode::Fill;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct Axis {
    line: [Vec3; 2],
    color: LinearRgba,
}

#[derive(Debug, Clone, Resource)]
struct RendererState {
    texts: Option<Vec<TextEmbedding>>,
    edges: Option<Vec<RenderedEdge>>,
}

#[derive(Resource, Default)]
struct EmbeddingEntities(pub std::collections::HashMap<String, Entity>);

#[derive(Resource, Default)]
struct EdgeEntities(pub std::collections::HashMap<String, Entity>);

#[derive(Resource)]
struct EmbeddingQueueResource;

/// Cache previous camera position to only update when moved
#[derive(Resource, Default)]
struct CameraCache {
    last_position: Option<Vec3>,
    frames_since_update: u32,
}

/// How often to update LOD/culling (every N frames)
const LOD_UPDATE_INTERVAL: u32 = 3;

// ---------------------------------------------------------------------------
// WASM API
// ---------------------------------------------------------------------------
#[wasm_bindgen]
pub fn add_embedding(x: f32, y: f32, z: f32) -> String {
    let embedding = JsEmbedding {
        id: Uuid::new_v4().to_string(),
        position: [x, y, z],
    };
    EMBEDDING_QUEUE.lock().unwrap().push(embedding.clone());
    embedding.id
}

#[wasm_bindgen]
pub fn add_edge(from_x: f32, from_y: f32, from_z: f32, to_x: f32, to_y: f32, to_z: f32) -> String {
    let edge = JsEdge {
        id: Uuid::new_v4().to_string(),
        from: [from_x, from_y, from_z],
        to: [to_x, to_y, to_z],
        color: None,
    };
    EDGE_QUEUE.lock().unwrap().push(edge.clone());
    edge.id
}

#[wasm_bindgen]
pub fn add_edge_with_color(
    from_x: f32,
    from_y: f32,
    from_z: f32,
    to_x: f32,
    to_y: f32,
    to_z: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
) -> String {
    let edge = JsEdge {
        id: Uuid::new_v4().to_string(),
        from: [from_x, from_y, from_z],
        to: [to_x, to_y, to_z],
        color: Some([r, g, b, a]),
    };
    EDGE_QUEUE.lock().unwrap().push(edge.clone());
    edge.id
}

#[wasm_bindgen]
pub fn delete_edge_by_id(id: String) {
    EDGE_DELETE_QUEUE.lock().unwrap().push(id);
}

#[wasm_bindgen]
pub fn get_edges() -> JsValue {
    let export = RENDERED_EDGES.lock().unwrap();
    serde_wasm_bindgen::to_value(&*export).unwrap_or(JsValue::NULL)
}

#[wasm_bindgen]
pub fn get_embeddings() -> JsValue {
    let export = RENDERED_EMBEDDINGS.lock().unwrap();
    serde_wasm_bindgen::to_value(&*export).unwrap_or(JsValue::NULL)
}

#[wasm_bindgen]
pub fn delete_embedding_by_id(id: String) {
    DELETE_QUEUE.lock().unwrap().push(id);
}

#[wasm_bindgen]
pub fn get_selected_id() -> Option<String> {
    SELECTED_EMBEDDING_ID.lock().unwrap().clone()
}

#[wasm_bindgen]
pub fn clear_selection() {
    *SELECTED_EMBEDDING_ID.lock().unwrap() = None;
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = should_show_ui)]
    fn should_show_ui() -> bool;
}

// ---------------------------------------------------------------------------
// Mesh conversion
// ---------------------------------------------------------------------------
impl From<Axis> for Mesh {
    fn from(axis: Axis) -> Self {
        let vertices = axis.line.to_vec();
        Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD)
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------
fn process_new_embeddings(mut renderer_state: ResMut<RendererState>) {
    match EMBEDDING_QUEUE.try_lock() {
        Ok(mut queue) => {
            for embedding in queue.drain(..) {
                let text = TextEmbedding {
                    id: embedding.id,
                    embedding: embedding.position,
                    drawn: false,
                };
                if let Some(texts) = &mut renderer_state.texts {
                    texts.push(text);
                } else {
                    renderer_state.texts = Some(vec![text]);
                }
            }
        }
        Err(e) => {
            warn!("could not get lock... {:#?}", e);
        }
    };
}

fn process_new_edges(mut renderer_state: ResMut<RendererState>) {
    match EDGE_QUEUE.try_lock() {
        Ok(mut queue) => {
            for edge in queue.drain(..) {
                let rendered = RenderedEdge {
                    id: edge.id,
                    from: edge.from,
                    to: edge.to,
                    drawn: false,
                };
                if let Some(edges) = &mut renderer_state.edges {
                    edges.push(rendered);
                } else {
                    renderer_state.edges = Some(vec![rendered]);
                }
            }
        }
        Err(e) => {
            warn!("could not get edge lock... {:#?}", e);
        }
    };
}

fn process_delete_embedding_requests(
    mut renderer_state: ResMut<RendererState>,
    mut entity_map: ResMut<EmbeddingEntities>,
    mut commands: Commands,
) {
    let mut queue = DELETE_QUEUE.lock().unwrap();
    if let Some(texts) = &mut renderer_state.texts {
        for id in queue.drain(..) {
            if let Some(entity) = entity_map.0.remove(&id) {
                commands.entity(entity).despawn_recursive();
            }
            texts.retain(|t| t.id != id);
        }
    }
}

fn process_delete_edge_requests(
    mut renderer_state: ResMut<RendererState>,
    mut edge_map: ResMut<EdgeEntities>,
    mut commands: Commands,
) {
    let mut queue = EDGE_DELETE_QUEUE.lock().unwrap();
    if let Some(edges) = &mut renderer_state.edges {
        for id in queue.drain(..) {
            if let Some(entity) = edge_map.0.remove(&id) {
                commands.entity(entity).despawn_recursive();
            }
            edges.retain(|e| e.id != id);
        }
    }
}

fn startup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
) {
    // Create shared meshes for LOD levels
    let sphere_high = meshes.add(
        Sphere::new(0.01)
            .mesh()
            .ico(SPHERE_SEGMENTS_HIGH as usize)
            .unwrap(),
    );
    let sphere_med = meshes.add(
        Sphere::new(0.01)
            .mesh()
            .ico(SPHERE_SEGMENTS_MED as usize)
            .unwrap(),
    );
    let sphere_low = meshes.add(
        Sphere::new(0.01)
            .mesh()
            .ico(SPHERE_SEGMENTS_LOW as usize)
            .unwrap(),
    );

    commands.insert_resource(SharedMeshes {
        sphere_high,
        sphere_med,
        sphere_low,
    });

    let axis_rgba = LinearRgba::new(0.7, 0.7, 0.7, 1.0);

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(VOLUME_CENTER + Vec3::new(0.0, 1.0, 2.5))
                .looking_at(VOLUME_CENTER, Vec3::Y),
            ..default()
        },
        PanOrbitCamera {
            focus: VOLUME_CENTER,
            allow_upside_down: true,
            pitch_upper_limit: Some(FRAC_PI_2 - 0.01),
            pitch_lower_limit: Some(-FRAC_PI_2 + 0.01),
            orbit_sensitivity: 5.0,
            pan_sensitivity: 0.0,
            zoom_sensitivity: 0.22,
            orbit_smoothness: 0.06,
            pan_smoothness: 0.00,
            zoom_smoothness: 0.06,
            zoom_lower_limit: Some(0.5),
            zoom_upper_limit: Some(5.0),
            touch_enabled: true,
            touch_controls: TouchControls::OneFingerOrbit,
            ..default()
        },
    ));

    // Axes
    let origin = Vec3::ZERO;
    for (end, color) in [
        (Vec3::X * AXIS_LENGTH, axis_rgba),
        (Vec3::Y * AXIS_LENGTH, axis_rgba),
        (Vec3::Z * AXIS_LENGTH, axis_rgba),
    ] {
        let axis = Axis {
            line: [origin, end],
            color,
        };
        commands.spawn((
            MaterialMeshBundle {
                mesh: meshes.add(axis),
                material: line_materials.add(LineMaterial { color }),
                ..default()
            },
            NoFrustumCulling, // Axes should always be visible
        ));
    }
}

fn update(mut contexts: EguiContexts, renderer_state: Res<RendererState>) {
    if should_show_ui() {
        egui::Window::new("Options")
            .collapsible(true)
            .movable(false)
            .show(contexts.ctx_mut(), |ui| {
                let num_texts = renderer_state.texts.as_ref().map_or(0, |t| t.len());
                ui.label(format!("Rendering {} Text Embeddings.", num_texts));

                if ui.button("Add Random Text Embedding").clicked() {
                    let mut rng = rand::thread_rng();
                    EMBEDDING_QUEUE.lock().unwrap().push(JsEmbedding {
                        id: Uuid::new_v4().to_string(),
                        position: [
                            rng.gen_range(-1.0..1.0),
                            rng.gen_range(-1.0..1.0),
                            rng.gen_range(-1.0..1.0),
                        ],
                    });
                }
            });
    }
}

fn sync_rendered_embeddings_to_js(renderer_state: Res<RendererState>) {
    if let Some(ref texts) = renderer_state.texts {
        let mut export = RENDERED_EMBEDDINGS.lock().unwrap();
        *export = texts.clone();
    }
}

fn sync_rendered_edges_to_js(renderer_state: Res<RendererState>) {
    if let Some(ref edges) = renderer_state.edges {
        let mut export = RENDERED_EDGES.lock().unwrap();
        *export = edges.clone();
    }
}

/// Draw new embeddings with shared mesh (instancing-like behavior)
fn draw_new_embeddings(
    mut commands: Commands,
    mut renderer_state: ResMut<RendererState>,
    shared_meshes: Res<SharedMeshes>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut entity_map: ResMut<EmbeddingEntities>,
) {
    let emb_color = [1.0_f32, 1.0, 1.0, 1.0];

    if let Some(texts) = &mut renderer_state.texts {
        for text in texts {
            if !text.drawn {
                let material = materials.add(StandardMaterial {
                    base_color: Color::srgba(
                        emb_color[0],
                        emb_color[1],
                        emb_color[2],
                        emb_color[3],
                    ),
                    emissive: LinearRgba::new(
                        emb_color[0],
                        emb_color[1],
                        emb_color[2],
                        emb_color[3],
                    ),
                    alpha_mode: AlphaMode::Blend,
                    ..default()
                });

                let entity = commands
                    .spawn((
                        PbrBundle {
                            mesh: shared_meshes.sphere_high.clone(), // Start with high LOD
                            material,
                            transform: Transform::from_xyz(
                                text.embedding[0],
                                text.embedding[1],
                                text.embedding[2],
                            ),
                            ..default()
                        },
                        text.clone(),
                        LodLevel::High,
                        // Add AABB for frustum culling (sphere radius 0.01)
                        Aabb::from_min_max(Vec3::splat(-0.01), Vec3::splat(0.01)),
                    ))
                    .id();

                entity_map.0.insert(text.id.clone(), entity);
                text.drawn = true;
            }
        }
    }
}

/// Draw edges as 3D lines
fn draw_new_edges(
    mut commands: Commands,
    mut renderer_state: ResMut<RendererState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut edge_map: ResMut<EdgeEntities>,
) {
    let default_color = LinearRgba::new(0.5, 0.5, 0.5, 0.6);

    if let Some(edges) = &mut renderer_state.edges {
        for edge in edges {
            if !edge.drawn {
                let from = Vec3::new(edge.from[0], edge.from[1], edge.from[2]);
                let to = Vec3::new(edge.to[0], edge.to[1], edge.to[2]);

                let line_mesh =
                    Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD)
                        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![from, to]);

                let entity = commands
                    .spawn((
                        MaterialMeshBundle {
                            mesh: meshes.add(line_mesh),
                            material: line_materials.add(LineMaterial {
                                color: default_color,
                            }),
                            ..default()
                        },
                        EdgeEntity {
                            id: edge.id.clone(),
                        },
                    ))
                    .id();

                edge_map.0.insert(edge.id.clone(), entity);
                edge.drawn = true;
            }
        }
    }
}

/// Optimized visibility/LOD/alpha system - runs every N frames
fn update_embedding_visibility(
    camera_query: Query<&Transform, With<Camera>>,
    mut emb_query: Query<
        (
            &Transform,
            &Handle<StandardMaterial>,
            &mut Handle<Mesh>,
            &mut LodLevel,
            &mut Visibility,
        ),
        With<TextEmbedding>,
    >,
    mut materials: ResMut<Assets<StandardMaterial>>,
    shared_meshes: Res<SharedMeshes>,
    mut cache: ResMut<CameraCache>,
) {
    // Throttle updates
    cache.frames_since_update += 1;

    let camera_transform = camera_query.single();
    let camera_pos = camera_transform.translation;

    // Skip if camera hasn't moved significantly and not time for periodic update
    if let Some(last_pos) = cache.last_position {
        let moved = camera_pos.distance(last_pos) > 0.01;
        if !moved && cache.frames_since_update < LOD_UPDATE_INTERVAL {
            return;
        }
    }

    cache.last_position = Some(camera_pos);
    cache.frames_since_update = 0;

    for (emb_transform, emb_material, mut mesh_handle, mut lod, mut visibility) in
        emb_query.iter_mut()
    {
        let distance = camera_pos.distance(emb_transform.translation);

        // Distance culling - hide very distant points
        if distance > CULL_DISTANCE {
            *visibility = Visibility::Hidden;
            continue;
        } else if *visibility == Visibility::Hidden {
            *visibility = Visibility::Inherited;
        }

        // LOD switching
        let new_lod = if distance < LOD_HIGH_DISTANCE {
            LodLevel::High
        } else if distance < LOD_MED_DISTANCE {
            LodLevel::Medium
        } else {
            LodLevel::Low
        };

        if new_lod != *lod {
            *lod = new_lod;
            *mesh_handle = match new_lod {
                LodLevel::High => shared_meshes.sphere_high.clone(),
                LodLevel::Medium => shared_meshes.sphere_med.clone(),
                LodLevel::Low => shared_meshes.sphere_low.clone(),
            };
        }

        // Alpha fade based on distance
        let alpha = if distance <= FADE_START_DISTANCE {
            1.0
        } else if distance >= FADE_END_DISTANCE {
            MIN_ALPHA
        } else {
            let t = (distance - FADE_START_DISTANCE) / (FADE_END_DISTANCE - FADE_START_DISTANCE);
            1.0 - t * (1.0 - MIN_ALPHA)
        };

        if let Some(material) = materials.get_mut(emb_material) {
            material.base_color.set_alpha(alpha);
            material.emissive.set_alpha(alpha);
        }
    }
}

fn log_window(window_q: Query<&Window, With<PrimaryWindow>>) {
    let w = window_q.single();
    info!(
        "logical={}x{}, physical={}x{}, scale_factor={}",
        w.width(),
        w.height(),
        w.physical_width(),
        w.physical_height(),
        w.scale_factor()
    );
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------
const SELECTION_RADIUS: f32 = 0.03;
const TAP_TOLERANCE: f32 = 10.0;

#[derive(Resource, Default)]
struct TouchTapTracker {
    start_position: Option<Vec2>,
    touch_id: Option<u64>,
}

fn distance_ray_to_point(ray: &Ray3d, point: Vec3) -> f32 {
    (ray.origin - point).cross(ray.direction.as_vec3()).length()
}

fn select_at_position(
    position: Vec2,
    camera: &Camera,
    cam_xform: &GlobalTransform,
    q_embeddings: &mut Query<(Entity, &Transform, &TextEmbedding, Option<&Selected>)>,
    commands: &mut Commands,
) {
    let Some(ray) = camera.viewport_to_world(cam_xform, position) else {
        return;
    };

    let mut best: Option<(Entity, f32)> = None;

    for (entity, transform, _emb, _sel) in q_embeddings.iter_mut() {
        let center = transform.translation;
        let d = distance_ray_to_point(&ray, center);

        if d <= SELECTION_RADIUS {
            let along = (center - ray.origin).dot(*ray.direction).max(0.0);
            match best {
                None => best = Some((entity, along)),
                Some((_e, best_along)) if along < best_along => best = Some((entity, along)),
                _ => {}
            }
        }
    }

    if best.is_none() {
        for (entity, _, _, maybe_sel) in q_embeddings.iter_mut() {
            if maybe_sel.is_some() {
                commands.entity(entity).remove::<Selected>();
            }
        }
        *SELECTED_EMBEDDING_ID.lock().unwrap() = None;
        return;
    }

    let (winner, _) = best.unwrap();

    let mut winner_was_selected = false;
    for (entity, _, _, maybe_sel) in q_embeddings.iter_mut() {
        if entity == winner && maybe_sel.is_some() {
            winner_was_selected = true;
        }
    }

    if winner_was_selected {
        commands.entity(winner).remove::<Selected>();
        *SELECTED_EMBEDDING_ID.lock().unwrap() = None;
    } else {
        for (entity, _, _, maybe_sel) in q_embeddings.iter_mut() {
            if maybe_sel.is_some() && entity != winner {
                commands.entity(entity).remove::<Selected>();
            }
        }
        commands.entity(winner).insert(Selected);
        for (entity, _, emb, _) in q_embeddings.iter() {
            if entity == winner {
                *SELECTED_EMBEDDING_ID.lock().unwrap() = Some(emb.id.clone());
                break;
            }
        }
    }
}

fn select_embedding_on_click(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cam_q: Query<(&Camera, &GlobalTransform)>,
    mut q_embeddings: Query<(Entity, &Transform, &TextEmbedding, Option<&Selected>)>,
    mut commands: Commands,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let window = windows.single();
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };

    let (camera, cam_xform) = cam_q.single();
    select_at_position(
        cursor_pos,
        camera,
        cam_xform,
        &mut q_embeddings,
        &mut commands,
    );
}

fn select_embedding_on_tap(
    touches: Res<Touches>,
    mut tap_tracker: ResMut<TouchTapTracker>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cam_q: Query<(&Camera, &GlobalTransform)>,
    mut q_embeddings: Query<(Entity, &Transform, &TextEmbedding, Option<&Selected>)>,
    mut commands: Commands,
) {
    for touch in touches.iter_just_pressed() {
        if touches.iter().count() == 1 {
            tap_tracker.start_position = Some(touch.position());
            tap_tracker.touch_id = Some(touch.id());
        } else {
            tap_tracker.start_position = None;
            tap_tracker.touch_id = None;
        }
    }

    for touch in touches.iter_just_released() {
        if let (Some(start_pos), Some(tracked_id)) =
            (tap_tracker.start_position, tap_tracker.touch_id)
        {
            if touch.id() == tracked_id {
                let end_pos = touch.position();
                if start_pos.distance(end_pos) < TAP_TOLERANCE {
                    let Ok(window) = windows.get_single() else {
                        tap_tracker.start_position = None;
                        tap_tracker.touch_id = None;
                        continue;
                    };

                    if end_pos.x >= 0.0
                        && end_pos.x <= window.width()
                        && end_pos.y >= 0.0
                        && end_pos.y <= window.height()
                    {
                        let (camera, cam_xform) = cam_q.single();
                        select_at_position(
                            end_pos,
                            camera,
                            cam_xform,
                            &mut q_embeddings,
                            &mut commands,
                        );
                    }
                }
            }
        }
        tap_tracker.start_position = None;
        tap_tracker.touch_id = None;
    }

    if let (Some(start_pos), Some(tracked_id)) = (tap_tracker.start_position, tap_tracker.touch_id)
    {
        for touch in touches.iter() {
            if touch.id() == tracked_id && start_pos.distance(touch.position()) > TAP_TOLERANCE {
                tap_tracker.start_position = None;
                tap_tracker.touch_id = None;
            }
        }
    }
}

fn apply_selection_visuals(mut q: Query<(&mut Transform, Option<&Selected>), With<TextEmbedding>>) {
    for (mut transform, maybe_sel) in q.iter_mut() {
        transform.scale = if maybe_sel.is_some() {
            Vec3::splat(3.0)
        } else {
            Vec3::ONE
        };
    }
}

fn register_internal_shader(mut shaders: ResMut<Assets<Shader>>) {
    let src = include_str!("../assets/shaders/line_material.wgsl");
    let shader = Shader::from_wgsl(src, "embedded://line_material.wgsl");
    shaders.insert(&LINE_SHADER_HANDLE, shader);
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
#[wasm_bindgen(start)]
pub fn run() {
    install_gecko_wheel_fix("#vector-canvas");

    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Vector Renderer".to_string(),
                    canvas: Some("#vector-canvas".into()),
                    fit_canvas_to_parent: true,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            MaterialPlugin::<LineMaterial>::default(),
            PanOrbitCameraPlugin,
            EguiPlugin,
        ))
        .insert_resource(ClearColor(Color::NONE))
        .insert_resource(EmbeddingEntities::default())
        .insert_resource(EdgeEntities::default())
        .insert_resource(RendererState {
            texts: None,
            edges: None,
        })
        .insert_resource(EmbeddingQueueResource)
        .insert_resource(TouchTapTracker::default())
        .insert_resource(CameraCache::default())
        .add_systems(Startup, (register_internal_shader, startup, log_window))
        .add_systems(PreUpdate, (process_new_embeddings, process_new_edges))
        .add_systems(
            Update,
            (
                update,
                draw_new_embeddings,
                draw_new_edges,
                process_delete_embedding_requests,
                process_delete_edge_requests,
                select_embedding_on_click,
                select_embedding_on_tap,
                apply_selection_visuals,
            ),
        )
        .add_systems(
            PostUpdate,
            (
                update_embedding_visibility, // Optimized LOD/culling/alpha
                sync_rendered_embeddings_to_js,
                sync_rendered_edges_to_js,
            ),
        )
        .run();
}

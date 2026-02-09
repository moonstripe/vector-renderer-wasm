use core::f32;
use core::f32::consts::FRAC_PI_2;

use bevy::{
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    reflect::TypePath,
    render::{
        mesh::{MeshVertexBufferLayoutRef, PrimitiveTopology},
        render_asset::RenderAssetUsages,
        render_resource::{
            AsBindGroup, PolygonMode, RenderPipelineDescriptor, ShaderRef,
            SpecializedMeshPipelineError,
        },
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
// Firefox reports wheel events with deltaMode=1 (DOM_DELTA_LINE) while
// Chromium uses deltaMode=0 (DOM_DELTA_PIXEL). Winit passes these through
// as different enum variants, and bevy_panorbit_camera applies different
// scaling to each — resulting in ~2.5× faster zoom on Firefox.
//
// We fix this at the JS level by patching incoming wheel events *in-place*
// (via Object.defineProperty) before winit's own listener fires. This keeps
// all Rust-side sensitivity values browser-agnostic.
// See: https://github.com/moonstripe/indra_net/issues/6
// ---------------------------------------------------------------------------
#[wasm_bindgen(inline_js = "
export function install_gecko_wheel_fix(canvas_selector) {
    const PIXELS_PER_LINE = 20;
    const canvas = document.querySelector(canvas_selector);
    if (!canvas) return;

    canvas.addEventListener('wheel', function(e) {
        // deltaMode 1 = DOM_DELTA_LINE  (Firefox default)
        // deltaMode 0 = DOM_DELTA_PIXEL (Chrome default)
        if (e.deltaMode === 1) {
            Object.defineProperty(e, 'deltaMode', { value: 0 });
            Object.defineProperty(e, 'deltaX',    { value: e.deltaX * PIXELS_PER_LINE });
            Object.defineProperty(e, 'deltaY',    { value: e.deltaY * PIXELS_PER_LINE });
            Object.defineProperty(e, 'deltaZ',    { value: e.deltaZ * PIXELS_PER_LINE });
        }
    }, { capture: true });   // capture phase → fires before winit's handler
}
")]
extern "C" {
    fn install_gecko_wheel_fix(canvas_selector: &str);
}

const AXIS_LENGTH: f32 = 500.;

// Center of the [0,1] volume - camera will focus here
const VOLUME_CENTER: Vec3 = Vec3::new(0.5, 0.5, 0.5);

static EMBEDDING_QUEUE: Lazy<Mutex<Vec<JsEmbedding>>> = Lazy::new(|| Mutex::new(Vec::new()));
static DELETE_QUEUE: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(Vec::new()));
static EDGE_QUEUE: Lazy<Mutex<Vec<JsEdge>>> = Lazy::new(|| Mutex::new(Vec::new()));
static EDGE_DELETE_QUEUE: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(Vec::new()));
static RENDERED_EMBEDDINGS: Lazy<Mutex<Vec<TextEmbedding>>> = Lazy::new(|| Mutex::new(Vec::new()));
static RENDERED_EDGES: Lazy<Mutex<Vec<RenderedEdge>>> = Lazy::new(|| Mutex::new(Vec::new()));
static SELECTED_EMBEDDING_ID: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

// set to arbitrary handle, since this is the only "external" asset
pub const LINE_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(0xA3E0_9C7D_8B51_42C3_9F77_12AB_34CD_5678); // arbitrary hex

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
    pub color: Option<[f32; 4]>, // RGBA, optional - defaults to gray
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RenderedEdge {
    id: String,
    from: [f32; 3],
    to: [f32; 3],
    drawn: bool,
}

#[derive(Resource)]
struct EmbeddingQueueResource;

#[wasm_bindgen]
pub fn add_embedding(x: f32, y: f32, z: f32) -> String {
    let embedding = JsEmbedding {
        id: Uuid::new_v4().to_string(),
        position: [x, y, z],
    };

    EMBEDDING_QUEUE.lock().unwrap().push(embedding.clone());

    embedding.id
}

/// Add an edge (line) between two 3D positions
/// Returns the edge ID
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

/// Add an edge with a custom color (RGBA, each component 0.0-1.0)
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

/// Delete an edge by ID
#[wasm_bindgen]
pub fn delete_edge_by_id(id: String) {
    EDGE_DELETE_QUEUE.lock().unwrap().push(id);
}

/// Get all rendered edges
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

/// Get the currently selected embedding ID (if any)
#[wasm_bindgen]
pub fn get_selected_id() -> Option<String> {
    SELECTED_EMBEDDING_ID.lock().unwrap().clone()
}

/// Clear the selection
#[wasm_bindgen]
pub fn clear_selection() {
    *SELECTED_EMBEDDING_ID.lock().unwrap() = None;
}

#[derive(Component, Debug, Clone, Serialize, Deserialize)]
struct TextEmbedding {
    id: String,
    embedding: [f32; 3],
    drawn: bool,
}

#[derive(Component)]
struct Selected;

/// Component to mark edge entities
#[derive(Component)]
struct EdgeEntity {
    id: String,
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

impl From<Axis> for Mesh {
    fn from(axis: Axis) -> Self {
        let vertices = axis.line.to_vec();

        Mesh::new(
            // Line list: every pair is a start/end point
            PrimitiveTopology::LineList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    }
}

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
    // Light gray for axes - high contrast on dark background
    let axis_rgba = LinearRgba::new(0.7, 0.7, 0.7, 1.0);

    commands.spawn((
        Camera3dBundle {
            // Position camera to look at center of [0,1] volume
            transform: Transform::from_translation(VOLUME_CENTER + Vec3::new(0.0, 1.0, 2.5))
                .looking_at(VOLUME_CENTER, Vec3::Y),
            ..default()
        },
        PanOrbitCamera {
            // Focus on center of volume
            focus: VOLUME_CENTER,

            // turn off auto "uprighting"; avoids the snap you're calling "sticky up"
            allow_upside_down: true,

            // don't let pitch reach the pole (numerical hell lives there)
            pitch_upper_limit: Some(FRAC_PI_2 - 0.01), // ~+89°
            pitch_lower_limit: Some(-FRAC_PI_2 + 0.01), // ~−89°

            // Tuned for the [0,1] volume scene scale
            orbit_sensitivity: 5.0,
            pan_sensitivity: 0.0,
            zoom_sensitivity: 0.22,
            orbit_smoothness: 0.06,
            pan_smoothness: 0.00,
            zoom_smoothness: 0.06,

            // sensible zoom bounds for your scene scale
            zoom_lower_limit: Some(0.5),
            zoom_upper_limit: Some(5.0),

            // Enable touch controls: one finger orbit, two finger pinch to zoom
            touch_enabled: true,
            touch_controls: TouchControls::OneFingerOrbit,

            ..default()
        },
    ));

    // Define axes - starting from origin, extending along each axis
    let origin = Vec3::ZERO;
    let x_axis_end = Vec3::X * AXIS_LENGTH;
    let y_axis_end = Vec3::Y * AXIS_LENGTH;
    let z_axis_end = Vec3::Z * AXIS_LENGTH;

    let x_axis = Axis {
        line: [origin, x_axis_end],
        color: axis_rgba,
    };
    let y_axis = Axis {
        line: [origin, y_axis_end],
        color: axis_rgba,
    };
    let z_axis = Axis {
        line: [origin, z_axis_end],
        color: axis_rgba,
    };

    // Spawn X Axis
    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(x_axis),
        material: line_materials.add(LineMaterial {
            color: x_axis.color,
        }),
        ..default()
    });

    // Spawn Y Axis
    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(y_axis),
        material: line_materials.add(LineMaterial {
            color: y_axis.color,
        }),
        ..default()
    });

    // Spawn Z Axis
    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(z_axis),
        material: line_materials.add(LineMaterial {
            color: z_axis.color,
        }),
        ..default()
    });
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
                    let x = rng.gen_range(-1.0..1.0);
                    let y = rng.gen_range(-1.0..1.0);
                    let z = rng.gen_range(-1.0..1.0);

                    EMBEDDING_QUEUE.lock().unwrap().push(JsEmbedding {
                        id: Uuid::new_v4().to_string(),
                        position: [x, y, z],
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

fn draw_new_embeddings(
    mut commands: Commands,
    mut renderer_state: ResMut<RendererState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut entity_map: ResMut<EmbeddingEntities>,
) {
    // White for embeddings - high contrast on dark background
    let emb_color = [1.0_f32, 1.0, 1.0, 1.0];

    if let Some(texts) = &mut renderer_state.texts {
        for text in texts {
            if !text.drawn {
                let entity = commands
                    .spawn(PbrBundle {
                        mesh: meshes.add(Sphere::new(0.01).mesh()),
                        material: materials.add(StandardMaterial {
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
                        }),
                        transform: Transform::from_xyz(
                            text.embedding[0],
                            text.embedding[1],
                            text.embedding[2],
                        ),
                        ..default()
                    })
                    .insert(text.clone())
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
    // Default edge color - semi-transparent gray
    let default_color = LinearRgba::new(0.5, 0.5, 0.5, 0.6);

    if let Some(edges) = &mut renderer_state.edges {
        for edge in edges {
            if !edge.drawn {
                let from = Vec3::new(edge.from[0], edge.from[1], edge.from[2]);
                let to = Vec3::new(edge.to[0], edge.to[1], edge.to[2]);

                // Create line mesh
                let line_mesh =
                    Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD)
                        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![from, to]);

                let entity = commands
                    .spawn(MaterialMeshBundle {
                        mesh: meshes.add(line_mesh),
                        material: line_materials.add(LineMaterial {
                            color: default_color,
                        }),
                        ..default()
                    })
                    .insert(EdgeEntity {
                        id: edge.id.clone(),
                    })
                    .id();

                edge_map.0.insert(edge.id.clone(), entity);
                edge.drawn = true;
            }
        }
    }
}

fn blur_embeddings(
    camera_query: Query<&Transform, With<Camera>>,
    mut emb_query: Query<(&Transform, &Handle<StandardMaterial>), With<TextEmbedding>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let camera_transform = camera_query.single();

    for (&emb_transform, emb_material) in emb_query.iter_mut() {
        let depth = (camera_transform.translation - emb_transform.translation).length();

        let opacity = if depth >= 5.0 {
            0.1
        } else {
            1.0 - (depth / 5.0) * 0.9
        };

        if let Some(material) = materials.get_mut(emb_material) {
            material.base_color.set_alpha(opacity);
            material.emissive.set_alpha(opacity);
        }
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = should_show_ui)]
    fn should_show_ui() -> bool;
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

/// Selection radius in world units around each embedding center.
/// This is larger than the sphere mesh radius (0.01) to make selection practical.
const SELECTION_RADIUS: f32 = 0.03;

/// Tolerance for tap vs drag detection (in pixels)
const TAP_TOLERANCE: f32 = 10.0;

/// Resource to track touch start position for tap detection
#[derive(Resource, Default)]
struct TouchTapTracker {
    start_position: Option<Vec2>,
    touch_id: Option<u64>,
}

/// Returns the shortest distance from a ray to a point in space.
fn distance_ray_to_point(ray: &Ray3d, point: Vec3) -> f32 {
    // |(p0 - point) x d|, where p0 is ray origin and d is unit ray direction
    (ray.origin - point).cross(ray.direction.as_vec3()).length()
}

/// Helper function to perform selection at a given screen position
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

    // Find closest embedding within threshold
    let mut best: Option<(Entity, f32)> = None;

    for (entity, transform, _emb, _sel) in q_embeddings.iter_mut() {
        let center = transform.translation;
        let d = distance_ray_to_point(&ray, center);

        if d <= SELECTION_RADIUS {
            // Use distance along the ray to prefer the front-most hit
            let along = (center - ray.origin).dot(*ray.direction).max(0.0);
            match best {
                None => best = Some((entity, along)),
                Some((_e, best_along)) if along < best_along => best = Some((entity, along)),
                _ => {}
            }
        }
    }

    // If nothing is close enough, clear selection
    if best.is_none() {
        // remove Selected from all
        for (entity, _, _, maybe_sel) in q_embeddings.iter_mut() {
            if maybe_sel.is_some() {
                commands.entity(entity).remove::<Selected>();
            }
        }
        // Clear JS-accessible selection
        *SELECTED_EMBEDDING_ID.lock().unwrap() = None;
        return;
    }

    let (winner, _) = best.unwrap();

    // Toggle semantics: if the winner is already selected, unselect it. Otherwise, select it and unselect others.
    let mut winner_was_selected = false;
    for (entity, _, _, maybe_sel) in q_embeddings.iter_mut() {
        if entity == winner && maybe_sel.is_some() {
            winner_was_selected = true;
        }
    }

    if winner_was_selected {
        commands.entity(winner).remove::<Selected>();
        // Clear selection in JS-accessible static
        *SELECTED_EMBEDDING_ID.lock().unwrap() = None;
    } else {
        // clear others
        for (entity, _, _, maybe_sel) in q_embeddings.iter_mut() {
            if maybe_sel.is_some() && entity != winner {
                commands.entity(entity).remove::<Selected>();
            }
        }
        // set winner
        commands.entity(winner).insert(Selected);
        // Update JS-accessible static with selected ID
        for (entity, _, emb, _) in q_embeddings.iter() {
            if entity == winner {
                *SELECTED_EMBEDDING_ID.lock().unwrap() = Some(emb.id.clone());
                break;
            }
        }
    }
}

/// Casts a ray from the cursor and selects the nearest embedding within SELECTION_RADIUS.
fn select_embedding_on_click(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cam_q: Query<(&Camera, &GlobalTransform)>,
    mut q_embeddings: Query<(Entity, &Transform, &TextEmbedding, Option<&Selected>)>,
    mut commands: Commands,
) {
    // Only act on fresh click
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

/// Track touch start/end for tap detection and selection on mobile
fn select_embedding_on_tap(
    touches: Res<Touches>,
    mut tap_tracker: ResMut<TouchTapTracker>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cam_q: Query<(&Camera, &GlobalTransform)>,
    mut q_embeddings: Query<(Entity, &Transform, &TextEmbedding, Option<&Selected>)>,
    mut commands: Commands,
) {
    // Track new touch starts
    for touch in touches.iter_just_pressed() {
        // Only track single finger taps
        if touches.iter().count() == 1 {
            tap_tracker.start_position = Some(touch.position());
            tap_tracker.touch_id = Some(touch.id());
        } else {
            // Multi-touch, cancel tap tracking
            tap_tracker.start_position = None;
            tap_tracker.touch_id = None;
        }
    }

    // Check for touch end (tap completion)
    for touch in touches.iter_just_released() {
        if let (Some(start_pos), Some(tracked_id)) =
            (tap_tracker.start_position, tap_tracker.touch_id)
        {
            if touch.id() == tracked_id {
                let end_pos = touch.position();
                let distance = start_pos.distance(end_pos);

                // If finger didn't move much, treat as a tap
                if distance < TAP_TOLERANCE {
                    let Ok(window) = windows.get_single() else {
                        tap_tracker.start_position = None;
                        tap_tracker.touch_id = None;
                        continue;
                    };

                    // Check if touch is within window bounds
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

        // Clear tracker after any touch release
        tap_tracker.start_position = None;
        tap_tracker.touch_id = None;
    }

    // Cancel tap if finger moves too much during the touch
    if let (Some(start_pos), Some(tracked_id)) = (tap_tracker.start_position, tap_tracker.touch_id)
    {
        for touch in touches.iter() {
            if touch.id() == tracked_id {
                let current_pos = touch.position();
                if start_pos.distance(current_pos) > TAP_TOLERANCE {
                    // User is dragging, not tapping
                    tap_tracker.start_position = None;
                    tap_tracker.touch_id = None;
                }
            }
        }
    }
}

/// Visually reflects selection by scaling selected embeddings up, others to normal.
fn apply_selection_visuals(mut q: Query<(&mut Transform, Option<&Selected>), With<TextEmbedding>>) {
    for (mut transform, maybe_sel) in q.iter_mut() {
        if maybe_sel.is_some() {
            // match previous behavior: x3 when selected
            transform.scale = Vec3::splat(3.0);
        } else {
            transform.scale = Vec3::ONE;
        }
    }
}

fn register_internal_shader(mut shaders: ResMut<Assets<Shader>>) {
    // path is relative to this file; adjust if you move things
    let src = include_str!("../assets/shaders/line_material.wgsl");
    let shader = Shader::from_wgsl(src, "embedded://line_material.wgsl");
    shaders.insert(&LINE_SHADER_HANDLE, shader);
}

#[wasm_bindgen(start)]
pub fn run() {
    // Patch Firefox wheel events before Bevy/winit registers its own listeners.
    install_gecko_wheel_fix("#vector-canvas");

    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Vector Renderer Test".to_string(),
                    canvas: Some("#vector-canvas".into()),
                    fit_canvas_to_parent: true,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            MaterialPlugin::<LineMaterial>::default(),
            PanOrbitCameraPlugin,
            EguiPlugin,
            // bevy_mod_picking removed
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
                select_embedding_on_click, // Mouse selection handler
                select_embedding_on_tap,   // Touch/tap selection handler
                apply_selection_visuals,   // Visual scaling for selection
            ),
        )
        .add_systems(
            PostUpdate,
            (
                blur_embeddings,
                sync_rendered_embeddings_to_js,
                sync_rendered_edges_to_js,
            ),
        )
        .run();
}

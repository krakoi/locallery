<script lang="ts">
  import type { ImageItem } from '../shared/types';

  let { item }: { item: ImageItem } = $props();
  let failed = $state(false);
</script>

<div class="video-player">
  <!-- svelte-ignore a11y_media_has_caption (original videos have no application-generated caption tracks) -->
  <video
    controls
    playsinline
    preload="metadata"
    poster={`/api/images/${item.id}/preview`}
    src={`/api/images/${item.id}/original`}
    onerror={() => (failed = true)}
  >
    Your browser cannot play this video.
  </video>
  {#if failed}
    <p role="alert">
      This format cannot play in your browser. Use Open original to play it
      externally.
    </p>
  {/if}
</div>

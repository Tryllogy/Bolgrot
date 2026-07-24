<script setup>
import { onMounted } from "vue";
import { screen, goHome, init } from "./composables/useGame.js";
import HomeView from "./components/HomeView.vue";
import EditorBar from "./components/EditorBar.vue";
import PlayControls from "./components/PlayControls.vue";
import GameBoard from "./components/GameBoard.vue";

onMounted(init);
</script>

<template>
  <div class="wrap">
    <HomeView v-if="screen === 'home'" />

    <template v-else>
      <div class="topbar">
        <button class="ghost" @click="goHome">← Accueil</button>
        <span class="title-sm">Bolgrot</span>
        <span class="topbar-spacer"></span>
      </div>
      <EditorBar v-if="screen === 'editor'" />
      <PlayControls v-if="screen === 'play'" />
      <GameBoard />
    </template>
  </div>
</template>

<style scoped>
.wrap {
  max-width: 1000px;
  margin: 0 auto;
  padding: 56px 24px 72px;
  text-align: center;
}
.topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 24px;
}
.title-sm {
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text-strong);
}
.topbar-spacer {
  flex: 1;
}
</style>

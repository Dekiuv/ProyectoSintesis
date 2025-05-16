// Mostrar/ocultar el menú desplegable del usuario
function toggleDropdown() {
    document.getElementById("dropdownMenu").classList.toggle("hidden");
  }

  // Cierra el menú si se hace clic fuera del área
  document.addEventListener("click", function (e) {
    if (!document.querySelector(".usuario").contains(e.target)) {
      document.getElementById("dropdownMenu").classList.add("hidden");
    }
  });

  // Abrir el panel de amigos
  async function abrirAmigos(event) {
    document.body.classList.add("no-scroll");
    event.stopPropagation();
    const panel = document.getElementById("panel-amigos");
    panel.classList.remove("hidden");

    const res = await fetch("/amigos");
    const amigos = await res.json();

    const lista = document.getElementById("lista-amigos");
    lista.innerHTML = "";

    amigos.forEach(amigo => {
      const div = document.createElement("div");
      div.className = "amigo-item";
      div.innerHTML = `
        <img src="${amigo.avatar}" class="amigo-avatar" alt="avatar">
        <span class="amigo-nombre">${amigo.nombre}</span>
      `;
      lista.appendChild(div);
    });
  }

  function cerrarPanelAmigos() {
    document.body.classList.remove("no-scroll");
    document.getElementById("panel-amigos").classList.add("hidden");
  }

  // Buscador en el panel de amigos
  document.addEventListener("DOMContentLoaded", function () {
    const buscador = document.getElementById("buscador-amigos");
    if (buscador) {
      buscador.addEventListener("input", function () {
        const filtro = this.value.toLowerCase();
        const amigos = document.querySelectorAll(".amigo-item");

        amigos.forEach(amigo => {
          const nombre = amigo.textContent.toLowerCase();
          amigo.style.display = nombre.includes(filtro) ? "flex" : "none";
        });
      });
    }
  });